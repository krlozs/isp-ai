"""
=============================================================
  ISP AI SUPPORT SYSTEM — BACKEND PRINCIPAL
  FastAPI + GLM 4.7-Flash + MikroWisp + SmartOLT
=============================================================
  Archivo: main.py
  Descripción: Servidor principal, webhooks y orquestación
=============================================================

Instalación de dependencias:
  pip install fastapi uvicorn httpx redis sqlalchemy
              alembic psycopg2-binary zhipuai pydantic
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
from zhipuai import ZhipuAI
from openai import AsyncOpenAI

from prompts import (
    SYSTEM_PROMPT, PROMPT_SALUDO, PROMPT_CLIENTE_IDENTIFICADO,
    PROMPT_DIAGNOSTICO_RED, PROMPT_POST_REBOOT, PROMPT_TROUBLESHOOTING,
    PROMPT_ESCALADO_TECNICO, PROMPT_CSAT, PROMPT_CLIENTE_FRUSTRADO,
    PROMPT_FUERA_HORARIO, MENSAJE_TECNICO_WHATSAPP, ISP_CONFIG
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN CLIENTE IA (Z.AI CODING PLAN)
# ─────────────────────────────────────────────

glm_client = AsyncOpenAI(
    api_key=os.getenv("GLM_API_KEY"),
    # Endpoint ESPECÍFICO para Coding Tools según soporte
    base_url="https://api.z.ai/api/coding/paas/v4" 
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ISP AI Support System", version="1.0.0")

# Clientes de servicios externos
# glm_client = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))
GLM_API_KEY = os.getenv("GLM_API_KEY")
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
    nombre: Optional[str] = None
    plan: Optional[str] = None
    serial_ont: Optional[str] = None
    ticket_id: Optional[str] = None
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
    await redis_client.setex(
        f"session:{session.phone}",
        ISP_CONFIG["session_ttl_minutes"] * 60,
        session.model_dump_json()
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
    temperatura: float = 0.7
) -> str:
    """
    Llama a Z.AI Coding Plan usando OpenAI Client.
    """
    system = SYSTEM_PROMPT.format(**ISP_CONFIG)

    messages = [{"role": "system", "content": system}]
    messages.extend(session.historial[-10:])
    messages.append({"role": "user", "content": prompt})

    try:
        response = await glm_client.chat.completions.create(
            model="GLM-4.5-Air",  # Modelo permitido en Coding Plan
            messages=messages,
            temperature=temperatura,
            max_tokens=2000
        )

        # Agregamos este log temporal para ver qué está llegando
        logger.info(f"DEBUG Z.AI Response: {response.model_dump_json()}")
        
        reply = response.choices[0].message.content

        # --- AGREGA ESTA VALIDACIÓN ---
        # Si la IA devuelve vacío, None o solo espacios, usamos un mensaje por defecto
        if not reply or reply.strip() == "":
            logger.warning("La IA devolvió una respuesta vacía, usando fallback.")
            reply = "Lo siento, no pude generar una respuesta en este momento."
        # -------------------------------

        session.historial.append({"role": "user", "content": prompt})
        session.historial.append({"role": "assistant", "content": reply})

        return reply

    except Exception as e:
        logger.error(f"Error Z.AI Coding Plan: {e}")
        return "Disculpa, tuve un problema al procesar tu solicitud (IA Coding)."

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
    Crea un ticket de soporte en MikroWisp.
    Retorna el ID del ticket creado.
    """
    headers = {
        "Authorization": f"Bearer {MIKROWISP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "cliente_id": datos["cliente_id"],
        "asunto": datos["asunto"],
        "descripcion": datos["descripcion"],
        "prioridad": datos.get("prioridad", "media"),
        "categoria": datos.get("categoria", "soporte_tecnico"),
        "datos_tecnicos": {
            "serial_ont": datos.get("serial_ont"),
            "señal_dbm": datos.get("señal_dbm"),
            "estado_ont": datos.get("estado_ont"),
            "reboot_ejecutado": datos.get("reboot_ejecutado", False),
            "pasos_realizados": datos.get("pasos_realizados", []),
        }
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(
                f"{MIKROWISP_BASE}/tickets",
                json=payload,
                headers=headers
            )
            if r.status_code in (200, 201):
                return r.json().get("id") or r.json().get("ticket_id")
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
# INTEGRACIÓN: SMARTOLT API
# ─────────────────────────────────────────────

async def so_get_ont_status(serial: str) -> Optional[dict]:
    """Obtiene el estado actual de una ONT por serial"""
    headers = {"Authorization": f"Bearer {SMARTOLT_KEY}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(
                f"{SMARTOLT_BASE}/onts/{serial}",
                headers=headers
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"Error SmartOLT get_ont: {e}")
    return None


async def so_get_signal(serial: str) -> Optional[dict]:
    """Obtiene nivel de señal óptica de la ONT"""
    headers = {"Authorization": f"Bearer {SMARTOLT_KEY}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(
                f"{SMARTOLT_BASE}/onts/{serial}/signal",
                headers=headers
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"Error SmartOLT get_signal: {e}")
    return None


async def so_get_alarmas(olt_id: str) -> list:
    """Obtiene alarmas activas en un nodo/OLT"""
    headers = {"Authorization": f"Bearer {SMARTOLT_KEY}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(
                f"{SMARTOLT_BASE}/alarms",
                params={"olt_id": olt_id},
                headers=headers
            )
            if r.status_code == 200:
                return r.json().get("alarms", [])
        except Exception as e:
            logger.error(f"Error SmartOLT get_alarmas: {e}")
    return []


async def so_reboot_ont(serial: str) -> bool:
    """
    Ejecuta reinicio remoto de una ONT.
    Retorna True si el comando fue enviado exitosamente.
    """
    headers = {
        "Authorization": f"Bearer {SMARTOLT_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.post(
                f"{SMARTOLT_BASE}/onts/{serial}/reboot",
                headers=headers
            )
            return r.status_code in (200, 202)
        except Exception as e:
            logger.error(f"Error SmartOLT reboot: {e}")
    return False


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
            reply = await call_glm(prompt, session)
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
        session.nombre = cliente.get("nombre")
        session.serial_ont = cliente.get("serial_ont") or cliente.get("ont_serial")

        # Verificar estado de cuenta
        facturas = await mw_get_facturas(str(cliente.get("id")))
        saldo = facturas.get("total_pendiente", 0)
        estado_cuenta = "CORTADO_MORA" if saldo > 0 and cliente.get("estado") == "suspendido" else "ACTIVO"

        prompt = PROMPT_CLIENTE_IDENTIFICADO.format(
            nombre=session.nombre,
            plan=cliente.get("plan", "N/A"),
            estado_servicio=cliente.get("estado", "activo"),
            saldo=f"${saldo:,.0f}" if saldo > 0 else "$0",
            ultimo_ticket=cliente.get("ultimo_ticket", "Ninguno"),
            fecha_vencimiento=cliente.get("fecha_vencimiento", "N/A"),
            estado_cuenta=estado_cuenta
        )
        reply = await call_glm(prompt, session)

        if estado_cuenta == "CORTADO_MORA":
            session.fase = "FINALIZADO_MORA"
        else:
            session.fase = "DIAGNOSTICO_RED"

        await save_session(session)
        await wa_send_message(phone, reply)
        return

    # ── FASE: DIAGNÓSTICO DE RED ─────────────
    elif session.fase == "DIAGNOSTICO_RED":

        ont_status = None
        señal = None
        alarmas = []

        if session.serial_ont:
            ont_status = await so_get_ont_status(session.serial_ont)
            señal_data = await so_get_signal(session.serial_ont)
            señal = señal_data.get("rx_power") if señal_data else None
            olt_id = ont_status.get("olt_id") if ont_status else None
            if olt_id:
                alarmas = await so_get_alarmas(olt_id)

        clientes_afectados = len([a for a in alarmas if a.get("severity") in ("critical", "major")])
        tipo_falla = "MASIVO" if clientes_afectados > 3 else ("INDIVIDUAL" if ont_status and ont_status.get("status") == "offline" else "NINGUNO")

        prompt = PROMPT_DIAGNOSTICO_RED.format(
            serial_ont=session.serial_ont or "N/A",
            estado_ont=ont_status.get("status", "desconocido") if ont_status else "no_encontrado",
            señal_dbm=señal or "N/A",
            ultima_vez_online=ont_status.get("last_seen", "N/A") if ont_status else "N/A",
            alarmas_nodo=len(alarmas),
            clientes_afectados=clientes_afectados,
            tipo_falla=tipo_falla,
            problema_cliente=mensaje
        )
        reply = await call_glm(prompt, session)

        if tipo_falla == "MASIVO":
            session.fase = "FINALIZADO_MASIVO"
        elif tipo_falla == "INDIVIDUAL" and session.serial_ont:
            # Intentar reinicio remoto
            session.fase = "REBOOT_PENDIENTE"
            bg.add_task(ejecutar_reboot_y_verificar, phone, session.serial_ont, session)
        else:
            session.fase = "TROUBLESHOOTING"

        await save_session(session)
        await wa_send_message(phone, reply)
        return

    # ── FASE: TROUBLESHOOTING GUIADO ─────────
    elif session.fase == "TROUBLESHOOTING":

        # Detectar frustración
        if detectar_frustracion(mensaje):
            prompt = PROMPT_CLIENTE_FRUSTRADO.format(mensaje_cliente=mensaje)
        else:
            prompt = PROMPT_TROUBLESHOOTING.format(
                pasos_realizados=", ".join(session.pasos_realizados) or "Ninguno aún",
                respuesta_cliente=mensaje
            )

        reply = await call_glm(prompt, session)

        # Detectar si el LLM decidió escalar
        if necesita_escalado(reply):
            session.fase = "ESCALADO"

        # Detectar si el LLM indicó resolución
        if esta_resuelto(reply):
            session.fase = "CSAT"

        await save_session(session)
        await wa_send_message(phone, reply)
        return

    # ── FASE: ESCALADO A TÉCNICO ─────────────
    elif session.fase == "ESCALADO":

        ticket_id = await mw_crear_ticket({
            "cliente_id": session.contrato,
            "asunto": "Falla técnica - Requiere visita",
            "descripcion": f"Diagnóstico IA: ONT {session.serial_ont}. Pasos: {', '.join(session.pasos_realizados)}",
            "serial_ont": session.serial_ont,
            "reboot_ejecutado": session.reboot_ejecutado,
            "pasos_realizados": session.pasos_realizados,
        })

        session.ticket_id = ticket_id
        horario = extraer_horario(mensaje)

        prompt = PROMPT_ESCALADO_TECNICO.format(
            nombre_cliente=session.nombre,
            contrato=session.contrato,
            plan="Plan registrado",
            problema="Falla de conectividad",
            estado_ont="Offline o degradado",
            señal="Ver ticket",
            reboot_ejecutado="Sí" if session.reboot_ejecutado else "No",
            resultado_reboot="Sin éxito" if session.reboot_ejecutado else "No ejecutado",
            pasos_realizados=", ".join(session.pasos_realizados),
            horario_preferido=horario,
            numero_ticket=ticket_id or "Pendiente"
        )
        reply = await call_glm(prompt, session)

        # Notificar al técnico
        if ticket_id:
            msg_tecnico = MENSAJE_TECNICO_WHATSAPP.format(
                numero_ticket=ticket_id,
                nombre_cliente=session.nombre,
                direccion="Ver MikroWisp",
                telefono=phone,
                plan="Ver MikroWisp",
                problema="Falla de conectividad sin resolución remota",
                estado_ont="Offline/Degradado",
                señal="Ver SmartOLT",
                reboot="Sí" if session.reboot_ejecutado else "No",
                resultado_reboot="Sin éxito",
                pasos_realizados="\n".join([f"• {p}" for p in session.pasos_realizados]),
                horario=horario
            )
            await wa_send_message(TECNICO_WHATSAPP, msg_tecnico)

        session.fase = "ESPERANDO_TECNICO"
        await save_session(session)
        await wa_send_message(phone, reply)
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
        reply = await call_glm(prompt, session)
        await wa_send_buttons(phone, reply, [
            {"id": "csat_1", "title": "1️⃣ Muy malo"},
            {"id": "csat_3", "title": "3️⃣ Regular"},
            {"id": "csat_5", "title": "5️⃣ Excelente"},
        ])
        await save_session(session)
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
        await wa_send_message(phone, "No pude ejecutar el reinicio remoto en este momento. Voy a guiarte para que lo hagas manualmente.")
        return

    await wa_send_message(phone, "⚙️ Reiniciando tu equipo remotamente... Por favor espera 2 minutos sin tocar el router.")
    await asyncio.sleep(ISP_CONFIG["reboot_wait_seconds"])

    # Verificar estado post-reinicio
    ont_post = await so_get_ont_status(serial)
    señal_post = await so_get_signal(serial)

    estado_post = ont_post.get("status", "offline") if ont_post else "offline"
    señal_val = señal_post.get("rx_power") if señal_post else None

    señal_ok = (
        señal_val is not None and
        ISP_CONFIG["señal_maxima_dbm"] >= float(señal_val) >= ISP_CONFIG["señal_minima_dbm"]
    )

    prompt = PROMPT_POST_REBOOT.format(
        estado_ont_post=estado_post,
        señal_post=señal_val or "N/A"
    )
    reply = await call_glm(prompt, session)

    if estado_post == "online" and señal_ok:
        session.fase = "CSAT"
    else:
        session.fase = "ESCALADO"

    await save_session(session)
    await wa_send_message(phone, reply)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extraer_contrato(texto: str) -> Optional[str]:
    """Extrae número de contrato o cédula del texto del cliente"""
    import re
    numeros = re.findall(r'\b\d{6,12}\b', texto)
    return numeros[0] if numeros else None


def extraer_horario(texto: str) -> str:
    """Detecta preferencia de horario en el texto"""
    texto_lower = texto.lower()
    if any(p in texto_lower for p in ["mañana", "manana", "am", "8", "9", "10", "11"]):
        return "Mañana (8am - 12pm)"
    if any(p in texto_lower for p in ["tarde", "pm", "1", "2", "3", "4", "5"]):
        return "Tarde (1pm - 5pm)"
    return "A coordinar con el técnico"


def detectar_frustracion(texto: str) -> bool:
    """Detecta señales de frustración en el mensaje del cliente"""
    palabras_clave = [
        "molesto", "cansado", "harto", "terrible", "pésimo", "pesimo",
        "nunca funciona", "siempre falla", "qué malo", "que malo",
        "incompetentes", "inútiles", "inutiles", "horrible", "basura"
    ]
    texto_lower = texto.lower()
    return any(p in texto_lower for p in palabras_clave)


def necesita_escalado(reply: str) -> bool:
    """Detecta si el LLM indicó necesidad de escalar"""
    indicadores = ["enviar técnico", "visita técnica", "técnico de campo", "escalar", "programar visita"]
    return any(i in reply.lower() for i in indicadores)


def esta_resuelto(reply: str) -> bool:
    """Detecta si el LLM confirmó resolución del problema"""
    indicadores = ["problema resuelto", "servicio restaurado", "ya tienes conexión", "funcionando correctamente"]
    return any(i in reply.lower() for i in indicadores)


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
        if msg_type == "text":
            texto = msg["text"]["body"]
        elif msg_type == "interactive":
            # Respuesta a botón
            texto = msg["interactive"]["button_reply"]["id"].replace("csat_", "")
        else:
            return JSONResponse({"status": "tipo_no_soportado"})

        logger.info(f"📱 Mensaje de {phone}: {texto[:50]}")

        # Procesar en background para responder rápido a Meta
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
            reply = await call_glm(prompt, session)
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
