"""
=============================================================
  ISP AI SUPPORT SYSTEM — BACKEND PRINCIPAL
  FastAPI + GLM 4.5-Air (httpx directo) + MikroWisp + SmartOLT
=============================================================
  Archivo: main.py
  Descripción: Servidor principal, webhooks y orquestación
=============================================================

Instalación de dependencias:
  pip install fastapi uvicorn httpx redis sqlalchemy
              alembic psycopg2-binary pydantic
              python-dotenv celery

Ejecutar:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
=============================================================
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from celery import Celery

from prompts import (
    SYSTEM_PROMPT, PROMPT_SALUDO, PROMPT_CLIENTE_IDENTIFICADO,
    PROMPT_DIAGNOSTICO_RED, PROMPT_POST_REBOOT, PROMPT_TROUBLESHOOTING,
    PROMPT_ESCALADO_TECNICO, PROMPT_CSAT, PROMPT_CLIENTE_FRUSTRADO,
    PROMPT_FUERA_HORARIO, MENSAJE_TECNICO_WHATSAPP, ISP_CONFIG
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# Silenciar librerías ruidosas en DEBUG
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("uvicorn").setLevel(logging.INFO)

app = FastAPI(title="ISP AI Support System", version="1.0.0")

GLM_API_KEY = os.getenv("GLM_API_KEY")
GLM_BASE_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
redis_client: aioredis.Redis = None

MIKROWISP_BASE = os.getenv("MIKROWISP_API_URL")       # ej: https://tu-mikrowisp.com/api/v1
MIKROWISP_TOKEN = os.getenv("MIKROWISP_API_TOKEN")

SMARTOLT_BASE = os.getenv("SMARTOLT_API_URL")          # ej: https://app.smartolt.com/api
SMARTOLT_KEY = os.getenv("SMARTOLT_API_KEY")

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID_CLIENTES")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

TECNICO_WHATSAPP = os.getenv("TECNICO_WHATSAPP_NUMBER")  # ej: 573001234567

# ─────────────────────────────────────────────
# CONFIGURACIÓN CELERY  <--- PEGA EL CÓDIGO AQUÍ
# ─────────────────────────────────────────────

celery_app = Celery(
    "main",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
)

# ─────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        encoding="utf-8",
        decode_responses=True
    )
    logger.info("✅ ISP AI System iniciado correctamente")


@app.on_event("shutdown")
async def shutdown():
    await redis_client.close()


# ─────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────

class SessionState(BaseModel):
    """Estado de la sesión de conversación de un cliente"""
    phone: str
    fase: str = "IDENTIFICACION"          # Fase actual del flujo
    contrato: Optional[str] = None
    id_cliente: Optional[str] = None      # ID interno de MikroWisp
    nombre: Optional[str] = None
    plan: Optional[str] = None
    serial_ont: Optional[str] = None
    ip_cliente: Optional[str] = None      # IP del servicio para ping
    ticket_id: Optional[str] = None
    kpi_activo: Optional[str] = None      # KPI seleccionado actualmente
    datos_tecnicos: Optional[str] = None  # Resultados técnicos para el ticket
    destino_escalado: str = "TECNICO"     # TECNICO o NOC
    pasos_realizados: list = []
    reboot_ejecutado: bool = False
    historial: list = []                  # Historial de mensajes para el LLM
    created_at: str = ""
    updated_at: str = ""


# ─────────────────────────────────────────────
# GESTIÓN DE SESIONES (Redis)
# ─────────────────────────────────────────────

async def get_session(phone: str) -> SessionState:
    """Obtiene o crea la sesión de un cliente desde Redis"""
    data = await redis_client.get(f"session:{phone}")
    if data:
        return SessionState(**json.loads(data))

    session = SessionState(
        phone=phone,
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    await save_session(session)
    return session


async def save_session(session: SessionState):
    """Guarda la sesión en Redis con TTL de 30 minutos"""
    session.updated_at = datetime.now().isoformat()
    try:
        # Pydantic v2
        data = session.model_dump_json()
    except Exception:
        # Pydantic v1 fallback
        data = json.dumps(session.dict())
    await redis_client.setex(
        f"session:{session.phone}",
        ISP_CONFIG["session_ttl_minutes"] * 60,
        data
    )


async def clear_session(phone: str):
    """Limpia la sesión al cerrar el ticket"""
    await redis_client.delete(f"session:{phone}")


# ─────────────────────────────────────────────
# INTEGRACIÓN: GLM (Vía OpenAI Compatible / Z.AI)
# ─────────────────────────────────────────────

async def call_glm(
    prompt: str,
    session: SessionState,
    raw_user_message: str,
    temperatura: float = 0.7
) -> str:
    """
    Llama a Z.AI usando httpx directo (sin SDK openai/zhipuai).
    """
    system = SYSTEM_PROMPT.format(**ISP_CONFIG)

    messages = [{"role": "system", "content": system}]
    messages.extend(session.historial[-10:])
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "GLM-4.5-Air",
        "messages": messages,
        "temperature": temperatura,
        "max_tokens": 2000
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(GLM_BASE_URL, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            reply = data["choices"][0]["message"]["content"]

        if raw_user_message and isinstance(raw_user_message, str):
            session.historial.append({"role": "user", "content": raw_user_message})
            session.historial.append({"role": "assistant", "content": reply})

        return reply

    except Exception as e:
        import traceback
        logger.error(f"Error GLM: {e}")
        logger.error(f"Error GLM traceback: {traceback.format_exc()}")
        return "Disculpa, tuve un problema al procesar tu solicitud."

# ─────────────────────────────────────────────
# INTEGRACIÓN: MIKROWISP API
# ─────────────────────────────────────────────

async def mw_get_cliente(contrato: str) -> Optional[dict]:
    """
    Obtiene datos del cliente desde MikroWisp usando POST y JSON
    """
    
    # Asegúrate que MIKROWISP_BASE termine en /api/v1
    url = f"{MIKROWISP_BASE}/GetClientsDetails"
    
    # El payload JSON (Cuerpo de la petición)
    payload = {
        "token": MIKROWISP_TOKEN,
        "cedula": contrato  # Usamos 'cedula' para buscar por dni
    }
    
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Hacemos POST enviando el JSON en el body
            r = await client.post(url, json=payload, headers=headers)
            
            logger.info(f"MIKROWISP URL: {r.url}")
            logger.info(f"MIKROWISP Status: {r.status_code}")
            logger.info(f"MIKROWISP Response: {r.text}")

            if r.status_code == 200:
                data = r.json()
                
                # CORRECCIÓN AQUÍ: Usamos .get() para leer el diccionario, no .post()
                if data.get("estado") == "exito":
                    clientes = data.get("datos", [])
                    if clientes:
                        return clientes[0]
                
                logger.warning(f"Cliente no encontrado para ID: {contrato}")
                return None
            else:
                logger.error(f"Error MikroWisp HTTP {r.status_code}: {r.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error de conexión con MikroWisp: {e}")
            return None


async def mw_get_facturas(cliente_id: str) -> dict:
    """Verifica el estado de cuenta del cliente usando POST y JSON (GetInvoices)"""
    
    # Endpoint para facturas
    url = f"{MIKROWISP_BASE}/GetInvoices"
    
    # Payload según documentación
    # estado: 1 = No pagadas (Pendientes)
    payload = {
        "token": MIKROWISP_TOKEN,
        "idcliente": cliente_id,
        "estado": 1,  # 1 significa facturas NO PAGADAS
        "limit": 10   # Opcional: Traer solo las últimas 10 para no saturar
    }
    
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # CORRECCIÓN: Usar POST y enviar JSON
            r = await client.post(url, json=payload, headers=headers)
            
            logger.info(f"MIKROWISP Facturas URL: {r.url}")
            logger.info(f"MIKROWISP Facturas Status: {r.status_code}")
            # logger.info(f"MIKROWISP Facturas Response: {r.text}") # Descomenta para debug

            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"Error MikroWisp get_facturas: {e}")
    return {}


async def mw_crear_ticket(datos: dict) -> Optional[str]:
    """
    Crea un ticket de soporte en MikroWisp usando /NewTicket.
    Retorna el ID del ticket creado.
    """
    from datetime import date
    headers = {"Content-Type": "application/json"}
    payload = {
        "token":       MIKROWISP_TOKEN,
        "idcliente":   datos["cliente_id"],
        "dp":          datos.get("dp", 1),
        "asunto":      datos.get("asunto", "Ticket de soporte"),
        "solicitante": datos.get("solicitante", "ARIA Bot"),
        "fechavisita": datos.get("fechavisita", date.today().strftime("%Y-%m-%d")),
        "turno":       datos.get("turno", "MAÑANA"),
        "agendado":    datos.get("agendado", "VIA TELEFONICA"),
        "contenido":   datos.get("descripcion", ""),
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(
                f"{MIKROWISP_BASE}/NewTicket",
                json=payload,
                headers=headers
            )
            logger.info(f"MikroWisp NewTicket status: {r.status_code} - {r.text}")
            if r.status_code == 200:
                data = r.json()
                if data.get("estado") == "exito":
                    return str(data.get("id") or data.get("ticket_id") or data.get("idticket", ""))
                else:
                    logger.error(f"MikroWisp NewTicket error: {data.get('mensaje')}")
        except Exception as e:
            logger.error(f"Error MikroWisp crear_ticket: {e}")
    return None


async def mw_cerrar_ticket(ticket_id: str, resolucion: str):
    """Cierra un ticket en MikroWisp con la descripción de resolución"""
    headers = {
        "Authorization": f"Bearer {MIKROWISP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "estado": "cerrado",
        "resolucion": resolucion,
        "fecha_cierre": datetime.now().isoformat()
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.put(
                f"{MIKROWISP_BASE}/tickets/{ticket_id}",
                json=payload,
                headers=headers
            )
        except Exception as e:
            logger.error(f"Error MikroWisp cerrar_ticket: {e}")


# ─────────────────────────────────────────────
# INTEGRACIÓN: SMARTOLT API (Versión 2 pasos)
# ─────────────────────────────────────────────

async def _get_onu_external_id(serial: str) -> Optional[str]:
    """Paso 1: Obtiene el unique_external_id usando el Serial Number."""
    headers = {"X-Token": SMARTOLT_KEY} 
    
    # Nota: Asegúrate que SMARTOLT_BASE en .env NO termine con /
    url = f"{SMARTOLT_BASE}/api/onu/get_onus_details_by_sn/{serial}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(url, headers=headers)
            if r.status_code == 200:
                data = r.json()
                
                # --- CORRECCIÓN AQUÍ ---
                # La respuesta tiene la estructura {'onus': [...]}
                onus_list = data.get("onus")
                
                if onus_list and len(onus_list) > 0:
                    # Tomamos el ID del primer elemento de la lista
                    onu_id = onus_list[0].get("unique_external_id")
                    if onu_id:
                        logger.info(f"SmartOLT ID encontrado para SN {serial}: {onu_id}")
                        return onu_id
                    else:
                        logger.warning(f"Field unique_external_id missing in onu item for SN {serial}")
                else:
                    logger.warning(f"Empty onus list in response for SN {serial}")
                # -------------------------
            else:
                logger.error(f"Error SmartOLT get_onus_details_by_sn: {r.status_code} - {r.text}")
        except Exception as e:
            logger.error(f"Error SmartOLT get_external_id: {e}")
    return None

async def so_get_ont_status(serial: str) -> Optional[dict]:
    """Obtiene el estado actual de una ONT (Paso 2)"""
    onu_id = await _get_onu_external_id(serial)
    if not onu_id:
        return None

    headers = {"X-Token": SMARTOLT_KEY}
    url = f"{SMARTOLT_BASE}/api/onu/get_onu_status/{onu_id}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(url, headers=headers)
            logger.debug(f"[DEBUG get_onu_status] URL: {url}")
            logger.debug(f"[DEBUG get_onu_status] Status HTTP: {r.status_code}")
            logger.debug(f"[DEBUG get_onu_status] Response raw: {r.text}")
            if r.status_code == 200:
                data = r.json()
                logger.info(f"[DEBUG get_onu_status] Parsed: {data}")
                return data
            else:
                logger.error(f"Error SmartOLT get_onu_status: {r.status_code}")
        except Exception as e:
            logger.error(f"Error SmartOLT get_status: {e}")
    return None


async def so_get_signal(serial: str) -> Optional[dict]:
    """Obtiene nivel de señal óptica de la ONT (Paso 2)"""
    onu_id = await _get_onu_external_id(serial)
    if not onu_id:
        return None

    headers = {"X-Token": SMARTOLT_KEY}
    url = f"{SMARTOLT_BASE}/api/onu/get_onu_signal/{onu_id}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(url, headers=headers)
            logger.debug(f"[DEBUG get_onu_signal] URL: {url}")
            logger.debug(f"[DEBUG get_onu_signal] Status HTTP: {r.status_code}")
            logger.debug(f"[DEBUG get_onu_signal] Response raw: {r.text}")
            if r.status_code == 200:
                data = r.json()
                logger.info(f"[DEBUG get_onu_signal] Parsed: {data}")
                # El campo real es onu_signal_1490 (Rx) y onu_signal_1310 (Tx)
                logger.info(f"[DEBUG get_onu_signal] onu_signal_1490 (Rx): {data.get('onu_signal_1490')}")
                logger.info(f"[DEBUG get_onu_signal] onu_signal_1310 (Tx): {data.get('onu_signal_1310')}")
                logger.info(f"[DEBUG get_onu_signal] onu_signal calidad: {data.get('onu_signal')}")
                return data
            else:
                logger.error(f"Error SmartOLT get_signal: {r.status_code}")
        except Exception as e:
            logger.error(f"Error SmartOLT get_signal: {e}")
    return None


async def so_reboot_ont(serial: str) -> bool:
    """
    Ejecuta reinicio remoto de una ONT (Paso 2).
    Retorna True si el comando fue enviado exitosamente.
    """
    # Paso 1: Obtener ID
    onu_id = await _get_onu_external_id(serial)
    if not onu_id:
        return False

    headers = {"X-Token": SMARTOLT_KEY}
    # Nota: El endpoint es POST según tu curl
    url = f"{SMARTOLT_BASE}/api/onu/reboot/{onu_id}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.post(url, headers=headers)
            return r.status_code in (200, 202)
        except Exception as e:
            logger.error(f"Error SmartOLT reboot: {e}")
    return False


async def so_get_full_status(serial: str) -> Optional[str]:
    """Obtiene el full status info de la ONT (señal, historial, WAN, interfaces)"""
    onu_id = await _get_onu_external_id(serial)
    if not onu_id:
        return None
    headers = {"X-Token": SMARTOLT_KEY}
    url = f"{SMARTOLT_BASE}/api/onu/get_onu_full_status_info/{onu_id}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            r = await client.get(url, headers=headers)
            if r.status_code == 200:
                return r.json().get("full_status_info")
            else:
                logger.error(f"Error SmartOLT full_status: {r.status_code}")
        except Exception as e:
            logger.error(f"Error SmartOLT full_status: {e}")
    return None


def parsear_full_status(raw: str) -> dict:
    """Parsea el texto plano de full_status_info y extrae los campos más relevantes."""
    import re
    resultado = {}

    def extraer(patron, texto, default="N/D"):
        m = re.search(patron, texto)
        return m.group(1).strip() if m else default

    resultado["rx_power"]        = extraer(r"Rx optical power\(dBm\)\s*:\s*(.+)", raw)
    resultado["tx_power"]        = extraer(r"Tx optical power\(dBm\)\s*:\s*(.+)", raw)
    resultado["olt_rx_power"]    = extraer(r"OLT Rx ONT optical power\(dBm\)\s*:\s*(.+)", raw)
    resultado["temperatura"]     = extraer(r"Temperature\(C\)\s*:\s*(.+)", raw)
    resultado["run_state"]       = extraer(r"Run state\s*:\s*(.+)", raw)
    resultado["last_down_cause"] = extraer(r"Last down cause\s*:\s*(.+)", raw)
    resultado["last_up_time"]    = extraer(r"Last up time\s*:\s*(.+)", raw)
    resultado["last_down_time"]  = extraer(r"Last down time\s*:\s*(.+)", raw)
    resultado["online_duration"] = extraer(r"ONT online duration\s*:\s*(.+)", raw)
    resultado["wan_status"]      = extraer(r"IPv4 Connection status\s*:\s*(.+)", raw)
    resultado["ipv4_address"]    = extraer(r"IPv4 address\s*:\s*(.+)", raw)
    resultado["wan_type"]        = extraer(r"IPv4 access type\s*:\s*(.+)", raw)

    # Historial de caídas (últimas 3)
    downs = re.findall(r"DownTime\s*:\s*(.+?)\nDownCause\s*:\s*(.+)", raw)
    historial = "\n".join([f"  - {t.strip()} -> {c.strip()}" for t, c in downs[:3]])
    resultado["historial_caidas"] = historial if historial else "Sin caídas recientes"

    return resultado


def formatear_datos_tecnicos(parsed: dict, ip_cliente: str, ping_resultado: str, kpi: str) -> str:
    """Genera el texto formateado para el ticket y el mensaje al técnico/NOC."""
    kpi_labels = {
        "kpi_no_internet":    "Sin acceso a internet",
        "kpi_lento_todo":     "Internet lento en todos los dispositivos",
        "kpi_wifi_lento":     "WiFi lento",
        "kpi_lag":            "Lag en juegos online",
        "kpi_intermitente":   "Conexión intermitente / se corta",
        "kpi_dns":            "No carga páginas web",
        "kpi_wifi_no_aparece":"Red WiFi no aparece",
    }
    problema = kpi_labels.get(kpi, kpi)
    return (
        f"DIAGNOSTICO TECNICO AUTOMATICO - ARIA\n"
        f"{'=' * 40}\n"
        f"Problema reportado: {problema}\n\n"
        f"SENAL OPTICA\n"
        f"  Rx ONT (dBm):     {parsed.get('rx_power')}\n"
        f"  Tx ONT (dBm):     {parsed.get('tx_power')}\n"
        f"  Rx OLT (dBm):     {parsed.get('olt_rx_power')}\n"
        f"  Temperatura:      {parsed.get('temperatura')} C\n\n"
        f"ESTADO WAN\n"
        f"  Conexion:         {parsed.get('wan_status')}\n"
        f"  Tipo:             {parsed.get('wan_type')}\n"
        f"  IP cliente:       {parsed.get('ipv4_address')} / {ip_cliente}\n\n"
        f"HISTORIAL ONT\n"
        f"  Estado actual:    {parsed.get('run_state')}\n"
        f"  Ultima caida:     {parsed.get('last_down_time')}\n"
        f"  Causa:            {parsed.get('last_down_cause')}\n"
        f"  Ultima subida:    {parsed.get('last_up_time')}\n"
        f"  Tiempo online:    {parsed.get('online_duration')}\n\n"
        f"ULTIMAS CAIDAS\n{parsed.get('historial_caidas')}\n\n"
        f"PING AL CLIENTE ({ip_cliente})\n  {ping_resultado}\n"
    )


async def ejecutar_ping(ip: str) -> str:
    """Ejecuta ping desde el servidor al cliente y retorna resultado formateado."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ping", "-c", "4", "-W", "2", ip,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        output = stdout.decode()
        import re
        resumen = re.search(r"(\d+ packets transmitted.+)", output)
        rtt = re.search(r"rtt.+?=\s*(.+)", output)
        lineas = []
        if resumen:
            lineas.append(resumen.group(1).strip())
        if rtt:
            lineas.append(f"RTT: {rtt.group(1).strip()}")
        return "\n  ".join(lineas) if lineas else "Sin respuesta (host inalcanzable)"
    except Exception as e:
        logger.error(f"Error ping {ip}: {e}")
        return "No se pudo ejecutar el ping"


# ─────────────────────────────────────────────
# INTEGRACIÓN: WHATSAPP API
# ─────────────────────────────────────────────

async def wa_send_message(to: str, message: str):
    """Envía un mensaje de texto por WhatsApp Business API"""
    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code != 200:
                logger.error(f"Error WhatsApp send: {r.text}")
        except Exception as e:
            logger.error(f"Error WhatsApp: {e}")


async def wa_send_buttons(to: str, body: str, buttons: list):
    """Envía mensaje con botones interactivos (máx 3 botones)"""
    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": b["id"], "title": b["title"]}}
                    for b in buttons[:3]
                ]
            }
        }
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post(url, json=payload, headers=headers)
        except Exception as e:
            logger.error(f"Error WhatsApp buttons: {e}")

async def wa_send_list(to: str, header_text: str, body_text: str, sections: list, button_text: str = "Ver opciones"):
    """Envía una lista desplegable (hasta 10 opciones) a WhatsApp."""
    url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Construir el JSON de la lista
    action_sections = []
    for sec in sections:
        rows = []
        for row in sec.get("rows", []):
            rows.append({
                "id": row["id"],
                "title": row["title"],
                "description": row.get("description", "")
            })
        action_sections.append({
            "title": sec.get("title", ""),
            "rows": rows
        })

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "header": {
                "type": "text",
                "text": header_text
            },
            "body": {
                "text": body_text
            },
            "footer": {
                "text": "ARIA - Soporte Técnico"
            },
            "action": {
                "button": button_text,
                "sections": action_sections
            }
        }
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code != 200:
                logger.error(f"Error WhatsApp List: {r.text}")
        except Exception as e:
            logger.error(f"Error WhatsApp List: {e}")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extraer_señal_rx(señal_data: dict) -> Optional[float]:
    """Extrae el valor numérico de señal Rx desde onu_signal_1490."""
    if not señal_data:
        return None
    raw = señal_data.get("onu_signal_1490") or señal_data.get("onu_signal_value", "")
    try:
        import re
        match = re.search(r"(-?\d+\.?\d*)", str(raw))
        if match:
            val = float(match.group(1))
            return val if val != 0.0 else None
    except (ValueError, TypeError):
        pass
    return None


def extraer_contrato(texto: str) -> Optional[str]:
    """Extrae número de contrato o cédula del texto del cliente"""
    import re
    numeros = re.findall(r'\b\d{6,12}\b', texto)
    return numeros[0] if numeros else None


def extraer_horario(texto: str) -> str:
    """Detecta preferencia de horario. Retorna MAÑANA o TARDE para MikroWisp."""
    texto_lower = texto.lower()
    if any(p in texto_lower for p in ["mañana", "manana", "am", "8", "9", "10", "11"]):
        return "MAÑANA"
    if any(p in texto_lower for p in ["tarde", "pm", "1", "2", "3", "4", "5"]):
        return "TARDE"
    return "MAÑANA"


def detectar_frustracion(texto: str) -> bool:
    """Detecta señales de frustración en el mensaje del cliente"""
    palabras_clave = [
        "molesto", "cansado", "harto", "terrible", "pésimo", "pesimo",
        "nunca funciona", "siempre falla", "qué malo", "que malo",
        "incompetentes", "inútiles", "inutiles", "horrible", "basura"
    ]
    return any(p in texto.lower() for p in palabras_clave)


def necesita_escalado(reply: str) -> bool:
    """Detecta si el LLM indicó necesidad de escalar"""
    indicadores = ["enviar técnico", "visita técnica", "técnico de campo", "escalar", "programar visita"]
    return any(i in reply.lower() for i in indicadores)


def esta_resuelto(reply: str) -> bool:
    """Detecta si el LLM confirmó resolución del problema"""
    indicadores = ["problema resuelto", "servicio restaurado", "ya tienes conexión", "funcionando correctamente"]
    return any(i in reply.lower() for i in indicadores)


# ─────────────────────────────────────────────
# LÓGICA PRINCIPAL DEL FLUJO
# ─────────────────────────────────────────────

async def procesar_mensaje(phone: str, mensaje: str, bg: BackgroundTasks):
    """
    Orquestador principal del flujo de atención.
    Gestiona el estado de la conversación y llama
    a los servicios correspondientes según la fase.
    """
    session = await get_session(phone)

    # ── FASE: IDENTIFICACIÓN ─────────────────
    if session.fase == "IDENTIFICACION":

        if not session.historial:
            # Primer mensaje → saludo
            prompt = PROMPT_SALUDO.format(mensaje_cliente=mensaje)
            reply = await call_glm(prompt, session, mensaje)
            await save_session(session)
            await wa_send_message(phone, reply)
            return

        # Buscar contrato en el mensaje
        contrato = extraer_contrato(mensaje)
        if not contrato:
            await wa_send_message(phone, "No pude identificar tu número de contrato. ¿Podrías escribirlo nuevamente? (solo números)")
            return

        # Consultar MikroWisp
        cliente = await mw_get_cliente(contrato)
        if not cliente:
            await wa_send_message(phone, f"No encontré ningún contrato con el número *{contrato}*. Verifica el número o escribe tu cédula.")
            return

        # Guardar datos en sesión
        session.contrato = contrato
        session.id_cliente = str(cliente.get("id"))
        session.nombre = cliente.get("nombre")

        # --- LÓGICA PARA MÚLTIPLES SERVICIOS CON SN INDIVIDUAL ---
        servicios = cliente.get("servicios", [])
        lista_planes_detalle = []
        import re
        
        # Variable para guardar el PRIMER SN encontrado como "principal" (por si lo necesitamos para reinicios rápidos)
        serial_principal_encontrado = None

        # Recorremos CADA servicio por separado
        for serv in servicios:
            tipo = serv.get("tiposervicio", "General")
            nombre_plan = serv.get("perfil", "Sin Plan")
            estado = serv.get("status_user", "Desconocido")
            
            # 1. Intentamos buscar el SN ESPECÍFICO de este servicio
            sn_texto = ""
            smartolt_data = serv.get("smartolt", "")
            match = re.search(r's:2:"sn";s:\d+:"([^"]+)"', smartolt_data)
            
            if match:
                sn_extraido = match.group(1)
                # Añadimos el SN al texto del servicio
                sn_texto = f" [SN: {sn_extraido}]"
                
                # Guardamos el primero como "serial_ont" de la sesión (para compatibilidad con el código de reinicio actual)
                if not serial_principal_encontrado:
                    serial_principal_encontrado = sn_extraido
            
            # 2. Construimos la línea del texto con el SN incluido
            lista_planes_detalle.append(f"- {tipo}: {nombre_plan} (Estado: {estado}){sn_texto}")

        # Guardamos la lista completa
        session.plan = "\n".join(lista_planes_detalle) if lista_planes_detalle else "N/A"
        
        # Guardamos el serial principal (para funciones que esperan un solo serial)
        session.serial_ont = serial_principal_encontrado

        # Guardar IP del primer servicio para ping
        if servicios:
            session.ip_cliente = servicios[0].get("ip")
        # -------------------------------------------------------------

        # Verificar estado de cuenta (resto igual)
        facturas = await mw_get_facturas(str(cliente.get("id")))
        saldo = facturas.get("total_pendiente", 0)
        estado_cuenta = "CORTADO_MORA" if saldo > 0 and cliente.get("estado") == "suspendido" else "ACTIVO"

        prompt = PROMPT_CLIENTE_IDENTIFICADO.format(
            nombre=session.nombre,
            plan=session.plan, # Ahora tendrá los SNs embebidos
            estado_servicio=cliente.get("estado", "activo"),
            saldo=f"${saldo:,.0f}" if saldo > 0 else "$0",
            ultimo_ticket=cliente.get("ultimo_ticket", "Ninguno"),
            fecha_vencimiento=cliente.get("fecha_vencimiento", "N/A"),
            estado_cuenta=estado_cuenta
        )
        reply = await call_glm(prompt, session, mensaje)

        if estado_cuenta == "CORTADO_MORA":
            session.fase = "FINALIZADO_MORA"
        else:
            session.fase = "DIAGNOSTICO_RED"

        await save_session(session)
        await wa_send_message(phone, reply)
        return

        # ── FASE: DIAGNÓSTICO DE RED ─────────
    elif session.fase == "DIAGNOSTICO_RED":

        ont_status = None
        señal_data = None
        onu_status_str = "desconocido"

        if session.serial_ont:
            ont_status = await so_get_ont_status(session.serial_ont)
            señal_data = await so_get_signal(session.serial_ont)

        # Determinar estado real de la ONT
        if ont_status:
            onu_status_str = ont_status.get("onu_status", "Offline").lower()

        señal_rx = None
        if señal_data:
            señal_rx = extraer_señal_rx(señal_data)
            logger.info(f"[SEÑAL] Rx extraída: {señal_rx} dBm | Calidad: {señal_data.get('onu_signal')}")

        SEÑAL_MIN = ISP_CONFIG.get("señal_minima_dbm", -27.0)
        SEÑAL_MAX = ISP_CONFIG.get("señal_maxima_dbm", -8.0)
        # Solo se considera degradada si tenemos un valor real y está fuera de rango
        señal_degradada = señal_rx is not None and not (SEÑAL_MAX >= señal_rx >= SEÑAL_MIN)

        # ── ESCENARIO A: ONT OFFLINE → Guía manual, sin reboot remoto
        if onu_status_str in ("offline", "power fail", "los"):
            session.fase = "TROUBLESHOOTING_MANUAL"
            session.pasos_realizados = [f"ont_estado:{onu_status_str}"]
            await save_session(session)
            await wa_send_message(
                phone,
                f"He detectado que tu equipo no tiene comunicación con nuestra red "
                f"(Estado: *{onu_status_str.upper()}*).\n\n"
                f"Vamos a intentar resolverlo juntos. Por favor revisa lo siguiente:\n\n"
                f"1️⃣ ¿Las luces de tu equipo están encendidas?\n"
                f"2️⃣ ¿El cable de fibra (amarillo o verde) está bien conectado?\n"
                f"3️⃣ ¿Hubo algún corte de luz recientemente?\n\n"
                f"Responde *Sí* si todo parece normal, o *No* si hay algo raro."
            )
            return

        # ── ESCENARIO B: ONT ONLINE con señal degradada → Reboot remoto
        elif onu_status_str == "online" and señal_degradada and session.serial_ont:
            session.fase = "REBOOT_PENDIENTE"
            session.pasos_realizados = ["senal_degradada"]
            await save_session(session)
            bg.add_task(ejecutar_reboot_y_verificar, phone, session.serial_ont, session)
            await wa_send_message(
                phone,
                f"He detectado que tu equipo está conectado pero con señal óptica degradada "
                f"(*{señal_rx} dBm*). Esto puede causar lentitud o cortes.\n\n"
                f"⚙️ Voy a reiniciar tu equipo remotamente para intentar estabilizarlo. "
                f"Por favor espera *2 minutos* sin tocar el router."
            )
            return

        # ── ESCENARIO C: ONT ONLINE señal normal → Mostrar lista KPI
        else:
            session.fase = "TROUBLESHOOTING"
            session.pasos_realizados = []

            secciones_menu = [
                {
                    "title": "📉 Problemas de Velocidad",
                    "rows": [
                        {"id": "kpi_lento_todo",  "title": "🐌 Todo internet lento"},
                        {"id": "kpi_wifi_lento",  "title": "📶 Solo WiFi lento"},
                        {"id": "kpi_lag",         "title": "🎮 Lag en juegos"},
                    ]
                },
                {
                    "title": "🚫 Problemas de Conexión",
                    "rows": [
                        {"id": "kpi_no_internet",  "title": "🚫 No tengo internet"},
                        {"id": "kpi_intermitente", "title": "⚡ Se corta a veces"},
                        {"id": "kpi_dns",          "title": "🌐 No carga páginas"},
                    ]
                },
                {
                    "title": "🔧 Otros",
                    "rows": [
                        {"id": "kpi_wifi_no_aparece", "title": "👻 No aparece mi WiFi"},
                    ]
                }
            ]

            session.pasos_realizados.append("menu_desplegado")
            await save_session(session)
            await wa_send_list(
                phone,
                header_text="Diagnóstico de Fallas",
                body_text=(
                    "He revisado tu equipo y está conectado correctamente a nuestra red. "
                    "Selecciona el problema que estás experimentando:"
                ),
                sections=secciones_menu,
                button_text="Seleccionar Problema"
            )
            return

    # ── FASE: TROUBLESHOOTING MANUAL (ONT OFFLINE) ────────
    elif session.fase == "TROUBLESHOOTING_MANUAL":

        respuesta = mensaje.lower().strip()
        if any(p in respuesta for p in ["sí", "si", "yes", "normal", "bien", "todo bien"]):
            # El cliente dice que todo parece normal pero sigue offline → escalar técnico
            session.kpi_activo = "ont_offline_sin_causa_aparente"
            session.destino_escalado = "TECNICO"
            session.datos_tecnicos = (
                f"ONT reportada OFFLINE por el sistema.\n"
                f"Estado cliente al consultar: {', '.join(session.pasos_realizados)}\n"
                f"Cliente confirmó que luces y cables parecen normales.\n"
                f"Serial ONT: {session.serial_ont}\n"
                f"IP cliente: {session.ip_cliente}"
            )
            session.fase = "ESCALADO"
            await save_session(session)
            await procesar_mensaje(phone, mensaje, bg)
        else:
            # Hay algo raro → verificar si volvió online
            ont_post = await so_get_ont_status(session.serial_ont) if session.serial_ont else None
            estado_post = ont_post.get("onu_status", "Offline") if ont_post else "Offline"

            if estado_post.lower() == "online":
                session.fase = "CSAT"
                await save_session(session)
                await wa_send_message(
                    phone,
                    "¡Buenas noticias! Tu equipo acaba de volver a conectarse a nuestra red. "
                    "Por favor prueba tu internet. ¿Se resolvió el problema?"
                )
            else:
                # Sigue offline → escalar técnico
                session.kpi_activo = "ont_offline_confirmado"
                session.destino_escalado = "TECNICO"
                session.datos_tecnicos = (
                    f"ONT OFFLINE confirmado.\n"
                    f"Cliente reportó anomalías en luces/cables.\n"
                    f"Serial ONT: {session.serial_ont}\n"
                    f"IP cliente: {session.ip_cliente}"
                )
                session.fase = "ESCALADO"
                await save_session(session)
                await procesar_mensaje(phone, mensaje, bg)
        return

    # ── FASE: TROUBLESHOOTING (KPIs desde lista) ──────────
    elif session.fase == "TROUBLESHOOTING":

        if not mensaje.startswith("kpi_"):
            await wa_send_message(
                phone,
                "Por favor selecciona una opción de la lista para que pueda registrar tu falla correctamente. 🙏"
            )
            return

        session.pasos_realizados.append(mensaje)
        session.kpi_activo = mensaje

        # ── KPI: VELOCIDAD (lento_todo, wifi_lento, lag) → Reboot con explicación
        if mensaje in ("kpi_lento_todo", "kpi_wifi_lento", "kpi_lag"):
            session.destino_escalado = "TECNICO"
            if session.serial_ont:
                await wa_send_message(
                    phone,
                    "Para mejorar tu velocidad voy a reiniciar tu equipo remotamente. 🔄\n\n"
                    "Es normal hacerlo *1-2 veces por semana* para limpiar la memoria del "
                    "equipo y mantener la conexión estable, igual que reiniciar un celular.\n\n"
                    "⚙️ Ejecutando reinicio... Por favor espera *2 minutos* sin tocar el router."
                )
                session.fase = "REBOOT_PENDIENTE"
                await save_session(session)
                bg.add_task(ejecutar_reboot_y_verificar, phone, session.serial_ont, session)
            else:
                session.fase = "ESCALADO"
                session.datos_tecnicos = f"KPI: {mensaje}. Sin serial ONT disponible."
                await save_session(session)
                await procesar_mensaje(phone, mensaje, bg)
            return

        # ── KPI: NO INTERNET → Verificar estado ONT primero
        elif mensaje == "kpi_no_internet":
            session.destino_escalado = "TECNICO"
            ont_status = await so_get_ont_status(session.serial_ont) if session.serial_ont else None
            onu_status_str = ont_status.get("onu_status", "Offline") if ont_status else "Offline"

            if onu_status_str.lower() in ("power fail", "los", "offline"):
                # Estado crítico → escalar técnico directo
                session.datos_tecnicos = (
                    f"KPI: Sin internet.\n"
                    f"Estado ONT al verificar: {onu_status_str}\n"
                    f"Serial ONT: {session.serial_ont}\n"
                    f"IP cliente: {session.ip_cliente}"
                )
                session.fase = "ESCALADO"
                await save_session(session)
                await procesar_mensaje(phone, mensaje, bg)
            else:
                # ONT online pero sin internet → hacer 2 preguntas
                session.fase = "ESPERANDO_PREGUNTAS_NOINET"
                session.pasos_realizados.append(f"ont_status_al_kpi:{onu_status_str}")
                await save_session(session)
                await wa_send_buttons(
                    phone,
                    "Tu equipo aparece conectado a nuestra red pero sin internet. "
                    "Para ayudarte mejor: ¿Qué luces ves en tu equipo ahora mismo?",
                    [
                        {"id": "luces_ninguna",  "title": "Sin luces"},
                        {"id": "luces_roja",     "title": "Luz roja/parpadeando"},
                        {"id": "luces_normal",   "title": "Luces normales"},
                    ]
                )
            return

        # ── KPI: INTERMITENTE → Full status + ping + escalar técnico
        elif mensaje == "kpi_intermitente":
            session.destino_escalado = "TECNICO"
            await wa_send_message(
                phone,
                "Entendido. Voy a revisar los registros técnicos de tu equipo y hacer "
                "pruebas de conectividad. Esto puede tomar unos segundos... ⏳"
            )
            raw = await so_get_full_status(session.serial_ont) if session.serial_ont else None
            ping_res = await ejecutar_ping(session.ip_cliente) if session.ip_cliente else "IP no disponible"

            if raw:
                parsed = parsear_full_status(raw)
                session.datos_tecnicos = formatear_datos_tecnicos(parsed, session.ip_cliente or "N/D", ping_res, mensaje)
            else:
                session.datos_tecnicos = f"KPI: {mensaje}.\nPing: {ping_res}\nSerial: {session.serial_ont}"

            session.fase = "ESCALADO"
            await save_session(session)
            await procesar_mensaje(phone, mensaje, bg)
            return

        # ── KPI: DNS / NO CARGA PÁGINAS → Full status + ping + escalar NOC
        elif mensaje == "kpi_dns":
            session.destino_escalado = "NOC"
            await wa_send_message(
                phone,
                "Entendido. Voy a revisar el estado de tu conexión WAN y hacer "
                "pruebas de red. Un momento... ⏳"
            )
            raw = await so_get_full_status(session.serial_ont) if session.serial_ont else None
            ping_res = await ejecutar_ping(session.ip_cliente) if session.ip_cliente else "IP no disponible"

            if raw:
                parsed = parsear_full_status(raw)
                session.datos_tecnicos = formatear_datos_tecnicos(parsed, session.ip_cliente or "N/D", ping_res, mensaje)
            else:
                session.datos_tecnicos = f"KPI: {mensaje}.\nPing: {ping_res}\nSerial: {session.serial_ont}"

            session.fase = "ESCALADO"
            await save_session(session)
            await procesar_mensaje(phone, mensaje, bg)
            return

        # ── KPI: WIFI NO APARECE → Escalar NOC directo
        elif mensaje == "kpi_wifi_no_aparece":
            session.destino_escalado = "NOC"
            session.datos_tecnicos = (
                f"KPI: Red WiFi no aparece en dispositivos del cliente.\n"
                f"Serial ONT: {session.serial_ont}\n"
                f"IP cliente: {session.ip_cliente}\n"
                f"Requiere revisión remota de configuración WiFi por NOC."
            )
            session.fase = "ESCALADO"
            await save_session(session)
            await procesar_mensaje(phone, mensaje, bg)
            return

    # ── FASE: PREGUNTAS ADICIONALES KPI_NO_INTERNET ───────
    elif session.fase == "ESPERANDO_PREGUNTAS_NOINET":

        session.pasos_realizados.append(f"luces:{mensaje}")

        # Primera pregunta respondida (luces) → hacer segunda pregunta
        if mensaje.startswith("luces_"):
            await save_session(session)
            await wa_send_buttons(
                phone,
                "Gracias. Segunda pregunta: ¿Hubo algún corte de luz eléctrica antes de que se fuera el internet?",
                [
                    {"id": "corte_si", "title": "✅ Sí hubo corte"},
                    {"id": "corte_no", "title": "❌ No hubo corte"},
                ]
            )
            return

        # Segunda pregunta respondida (corte de luz) → escalar con contexto
        luces = next((p for p in session.pasos_realizados if p.startswith("luces:")), "luces:desconocido")
        corte = "Sí" if mensaje == "corte_si" else "No"

        session.datos_tecnicos = (
            f"KPI: Sin acceso a internet (ONT aparece online).\n"
            f"Luces del equipo: {luces.replace('luces:', '')}\n"
            f"Corte de luz previo: {corte}\n"
            f"Serial ONT: {session.serial_ont}\n"
            f"IP cliente: {session.ip_cliente}"
        )
        session.destino_escalado = "TECNICO"
        session.fase = "ESCALADO"
        await save_session(session)
        await procesar_mensaje(phone, mensaje, bg)
        return
      
    # ── FASE: ESCALADO A TÉCNICO / NOC ───────────────────
    elif session.fase == "ESCALADO":

        kpi_labels = {
            "kpi_no_internet":         "Sin acceso a internet",
            "kpi_lento_todo":          "Internet lento en todos los dispositivos",
            "kpi_wifi_lento":          "WiFi lento",
            "kpi_lag":                 "Lag en juegos online",
            "kpi_intermitente":        "Conexión intermitente / se corta",
            "kpi_dns":                 "No carga páginas web",
            "kpi_wifi_no_aparece":     "Red WiFi no aparece",
            "ont_offline_sin_causa_aparente": "ONT offline sin causa aparente",
            "ont_offline_confirmado":  "ONT offline confirmado por cliente",
        }
        problema_texto = kpi_labels.get(session.kpi_activo or "", "Falla de conectividad")
        horario = extraer_horario(mensaje)
        destino = session.destino_escalado or "TECNICO"
        reboot_texto = "Sí, sin éxito" if session.reboot_ejecutado else "No fue necesario"

        contenido_ticket = (
            f"Reporte generado por ARIA (Soporte IA)\n"
            f"{'=' * 40}\n"
            f"Problema reportado: {problema_texto}\n"
            f"Serial ONT: {session.serial_ont or 'No disponible'}\n"
            f"IP cliente: {session.ip_cliente or 'No disponible'}\n"
            f"Reinicio remoto: {reboot_texto}\n"
            f"Atendido via: WhatsApp\n"
            f"Telefono: {phone}\n\n"
        )
        if session.datos_tecnicos:
            contenido_ticket += session.datos_tecnicos

        ticket_id = await mw_crear_ticket({
            "cliente_id":  session.id_cliente,
            "asunto":      f"Falla tecnica: {problema_texto[:50]}",
            "descripcion": contenido_ticket,
            "solicitante": session.nombre or "Cliente",
            "turno":       horario,
            "agendado":    "VIA TELEFONICA",
        })

        session.ticket_id = ticket_id
        numero_destino = os.getenv("NOC_WHATSAPP") if destino == "NOC" else TECNICO_WHATSAPP
        logger.info(f"Ticket creado: #{ticket_id} | Destino: {destino} | Número notificación: {numero_destino}")

        # Mensaje al cliente
        await wa_send_message(
            phone,
            f"He registrado tu caso con el ticket *#{ticket_id}*. 📋\n\n"
            f"Un {'técnico' if destino == 'TECNICO' else 'especialista'} revisará tu caso "
            f"y se pondrá en contacto contigo a la brevedad.\n\n"
            f"Si tienes alguna consulta adicional puedes escribirnos aquí. 🙏"
        )

        # Notificar al técnico o NOC por WhatsApp
        if ticket_id and numero_destino:
            msg_destino = (
                f"🔔 *NUEVO TICKET #{ticket_id}*\n"
                f"{'=' * 30}\n"
                f"👤 Cliente: {session.nombre}\n"
                f"📋 Contrato: {session.contrato}\n"
                f"📱 Teléfono: {phone}\n"
                f"🔌 Serial ONT: {session.serial_ont or 'N/D'}\n"
                f"🌐 IP: {session.ip_cliente or 'N/D'}\n"
                f"⚠️ Problema: {problema_texto}\n"
                f"🔄 Reboot remoto: {reboot_texto}\n"
                f"🕐 Horario preferido: {horario}\n\n"
            )
            if session.datos_tecnicos:
                msg_destino += f"📊 *DIAGNÓSTICO:*\n{session.datos_tecnicos}"
            await wa_send_message(numero_destino, msg_destino)

        session.fase = "ESPERANDO_TECNICO"
        await save_session(session)
        return

    # ── FASE: ENCUESTA CSAT ──────────────────
    elif session.fase == "CSAT":

        # Si recibimos la calificación (1-5)
        if mensaje.strip() in ["1", "2", "3", "4", "5"]:
            calificacion = int(mensaje.strip())
            # TODO: Guardar en PostgreSQL
            logger.info(f"CSAT recibido: {calificacion} - Cliente: {phone}")
            if session.ticket_id:
                await mw_cerrar_ticket(session.ticket_id, f"Resuelto. CSAT: {calificacion}/5")
            await wa_send_message(phone, f"¡Gracias por tu calificación! {'⭐' * calificacion}\n\nTu opinión nos ayuda a mejorar. ¡Hasta pronto! 👋")
            await clear_session(phone)
            return

        prompt = PROMPT_CSAT.format(
            nombre_cliente=session.nombre,
            tipo_resolucion="REMOTA",
            tiempo_resolucion="Pocos minutos"
        )
        reply = await call_glm(prompt, session, mensaje)
        await wa_send_buttons(phone, reply, [
            {"id": "csat_1", "title": "1️⃣ Muy malo"},
            {"id": "csat_3", "title": "3️⃣ Regular"},
            {"id": "csat_5", "title": "5️⃣ Excelente"},
        ])
        await save_session(session)
        return

    # ── FASE: ESPERANDO TÉCNICO ──────────────────────────
    elif session.fase == "ESPERANDO_TECNICO":
        prompt = (
            f"El cliente {session.nombre} tiene el ticket #{session.ticket_id} activo y está esperando "
            f"la visita o atención del {'técnico' if session.destino_escalado == 'TECNICO' else 'equipo NOC'}. "
            f"Ahora pregunta: '{mensaje}'. "
            f"Responde amablemente, confirma que su ticket está registrado, NO prometas horarios específicos "
            f"y anímalo a tener paciencia. Sé breve."
        )
        reply = await call_glm(prompt, session, mensaje)
        await wa_send_message(phone, reply)
        return

    # ── FASE: DEFAULT ─────────────────────────
    else:
        await wa_send_message(phone, "Tu caso está siendo atendido. Si tienes alguna consulta adicional, escríbenos. 🙏")


# ─────────────────────────────────────────────
# TAREA DE FONDO: REBOOT + VERIFICACIÓN
# ─────────────────────────────────────────────

async def ejecutar_reboot_y_verificar(phone: str, serial: str, session: SessionState):
    """
    Ejecuta el reinicio remoto de la ONT y verifica
    el resultado después de 2 minutos.
    """
    exito = await so_reboot_ont(serial)
    session.reboot_ejecutado = True

    if not exito:
        session.fase = "TROUBLESHOOTING"
        await save_session(session)
        await wa_send_message(phone, "No pude ejecutar el reinicio remoto en este momento. Por favor intenta apagar y encender tu equipo manualmente, espera 2 minutos y escríbenos si el problema persiste.")
        return

    await wa_send_message(phone, "⚙️ Reiniciando tu equipo remotamente... Por favor espera 2 minutos sin tocar el router.")
    await asyncio.sleep(ISP_CONFIG.get("reboot_wait_seconds", 120))

    # Verificar estado post-reinicio
    ont_post = await so_get_ont_status(serial)
    señal_post = await so_get_signal(serial)

    estado_post = ont_post.get("onu_status", "Offline").lower() if ont_post else "offline"
    señal_val = extraer_señal_rx(señal_post) if señal_post else None
    logger.info(f"[POST-REBOOT SEÑAL] Rx: {señal_val} dBm")

    SEÑAL_MIN = ISP_CONFIG.get("señal_minima_dbm", -27.0)
    SEÑAL_MAX = ISP_CONFIG.get("señal_maxima_dbm", -8.0)
    señal_ok = señal_val is not None and (SEÑAL_MAX >= señal_val >= SEÑAL_MIN)

    if estado_post == "online" and señal_ok:
        session.fase = "CSAT"
        await save_session(session)
        await wa_send_message(
            phone,
            f"✅ ¡Tu equipo se reinició correctamente y la señal está estable ({señal_val} dBm)!\n\n"
            f"Por favor prueba tu internet. ¿Se resolvió el problema?"
        )
    else:
        # Escalar directamente sin llamar procesar_mensaje (bg no disponible en background task)
        session.fase = "ESPERANDO_TECNICO"
        session.destino_escalado = "TECNICO"
        session.datos_tecnicos = (
            f"Reboot remoto ejecutado.\n"
            f"Estado post-reboot: {estado_post}\n"
            f"Señal post-reboot: {señal_val or 'N/D'} dBm\n"
            f"KPI original: {session.kpi_activo or 'velocidad/sin_internet'}"
        )

        kpi_labels = {
            "kpi_lento_todo": "Internet lento",
            "kpi_wifi_lento": "WiFi lento",
            "kpi_lag":        "Lag en juegos",
            "senal_degradada":"Señal óptica degradada",
        }
        problema_texto = kpi_labels.get(session.kpi_activo or "", "Falla de conectividad post-reboot")

        ticket_id = await mw_crear_ticket({
            "cliente_id":  session.id_cliente,
            "asunto":      f"Falla técnica: {problema_texto[:50]}",
            "descripcion": session.datos_tecnicos,
            "solicitante": session.nombre or "Cliente",
            "turno":       "MAÑANA",
            "agendado":    "VIA TELEFONICA",
        })
        session.ticket_id = ticket_id
        await save_session(session)

        await wa_send_message(
            phone,
            f"El reinicio se ejecutó pero tu equipo no logró estabilizarse. "
            f"He registrado tu caso con el ticket *#{ticket_id}*. 🔧\n\n"
            f"Un técnico revisará tu caso y se pondrá en contacto contigo a la brevedad."
        )

        if ticket_id and TECNICO_WHATSAPP:
            msg_tecnico = (
                f"🔔 *NUEVO TICKET #{ticket_id}* (Post-Reboot)\n"
                f"{'=' * 30}\n"
                f"👤 Cliente: {session.nombre}\n"
                f"📋 Contrato: {session.contrato}\n"
                f"📱 Teléfono: {phone}\n"
                f"🔌 Serial ONT: {session.serial_ont or 'N/D'}\n"
                f"🌐 IP: {session.ip_cliente or 'N/D'}\n"
                f"⚠️ Problema: {problema_texto}\n"
                f"🔄 Reboot remoto: Sí, sin éxito\n"
                f"📊 Estado post-reboot: {estado_post} | Señal: {señal_val or 'N/D'} dBm"
            )
            await wa_send_message(TECNICO_WHATSAPP, msg_tecnico)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
# WEBHOOK WHATSAPP
# ─────────────────────────────────────────────

@app.get("/webhook")
async def verificar_webhook(request: Request):
    """Verificación del webhook por Meta"""
    from fastapi.responses import PlainTextResponse
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge, status_code=200)
    raise HTTPException(status_code=403, detail="Token inválido")


@app.post("/webhook")
async def recibir_mensaje(request: Request, bg: BackgroundTasks):
    """
    Webhook principal. Recibe mensajes de WhatsApp
    y los despacha al procesador del flujo.
    """
    try:
        body = await request.json()
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})

        messages = value.get("messages", [])
        if not messages:
            return JSONResponse({"status": "no_messages"})

        msg = messages[0]
        phone = msg.get("from")
        msg_type = msg.get("type")

        # Extraer texto según tipo de mensaje
        texto = None
        if msg_type == "text":
            texto = msg["text"]["body"]
        elif msg_type == "interactive":
            interactive_data = msg.get("interactive", {})
            tipo_interactivo = interactive_data.get("type")
            
            if tipo_interactivo == "list_reply":
                list_reply_id = interactive_data["list_reply"]["id"]
                texto = list_reply_id
                logger.info(f"KPI SELECCIONADO: {list_reply_id}")

            elif tipo_interactivo == "button_reply":
                button_id = interactive_data["button_reply"]["id"]
                texto = button_id.replace("csat_", "")
            
            else:
                return JSONResponse({"status": "interactive_type_not_supported"})

        if not texto:
            return JSONResponse({"status": "no_text"})

        logger.info(f"📱 Mensaje de {phone}: {texto[:50]}")
        bg.add_task(procesar_mensaje, phone, texto, bg)
        return JSONResponse({"status": "ok"})

    except Exception as e:
        logger.error(f"Error en webhook: {e}")
        return JSONResponse({"status": "error"}, status_code=200)


# ─────────────────────────────────────────────
# WEBHOOK MIKROWISP — Cierre de tickets
# ─────────────────────────────────────────────

@app.post("/webhook/mikrowisp/ticket-closed")
async def ticket_cerrado_por_tecnico(request: Request):
    """
    Webhook de MikroWisp. Se dispara cuando el técnico
    actualiza un ticket a 'resuelto' desde campo.
    """
    try:
        data = await request.json()
        ticket_id = data.get("ticket_id")
        cliente_phone = data.get("cliente_telefono")

        if cliente_phone:
            session = await get_session(cliente_phone)
            session.fase = "CSAT"
            await save_session(session)

            prompt = PROMPT_CSAT.format(
                nombre_cliente=session.nombre or "cliente",
                tipo_resolucion="VISITA_TECNICA",
                tiempo_resolucion="Visita técnica completada"
            )
            reply = await call_glm(prompt, session, mensaje)
            await wa_send_message(cliente_phone, reply)

        return JSONResponse({"status": "ok"})
    except Exception as e:
        logger.error(f"Error webhook MikroWisp: {e}")
        return JSONResponse({"status": "error"})


# ─────────────────────────────────────────────
# ENDPOINT DE SALUD
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": "ok",
            "glm": "ok",
            "mikrowisp": MIKROWISP_BASE is not None,
            "smartolt": SMARTOLT_BASE is not None,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
