"""
=============================================================
  ISP AI SUPPORT SYSTEM — MÓDULO TÉCNICO
  Gestión autónoma de la conversación con el técnico
=============================================================
  Archivo: tecnico.py
  Descripción: El IA gestiona toda la interacción con el
               técnico por WhatsApp y actualiza MikroWisp
               automáticamente. El técnico NUNCA entra
               al sistema, solo usa WhatsApp.
=============================================================
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PROMPTS DEL TÉCNICO
# ─────────────────────────────────────────────

PROMPT_BRIEF_TECNICO = """
Redacta el mensaje de asignación de ticket para el técnico de campo.
El técnico lo recibirá por WhatsApp. Debe ser claro, directo y con toda
la información que necesita para la visita. Usa emojis para que sea fácil
de leer en el celular.

DATOS DEL CASO:
- Ticket #: {ticket_id}
- Cliente: {nombre_cliente}
- Dirección: {direccion}
- Teléfono cliente: {telefono_cliente}
- Plan: {plan}
- Problema reportado: {problema}
- Estado ONT: {estado_ont}
- Señal óptica: {señal_dbm} dBm (óptimo: -8 a -27 dBm)
- Reboot remoto ejecutado: {reboot_ejecutado}
- Resultado del reboot: {resultado_reboot}
- Diagnóstico previo del IA: {diagnostico_ia}
- Horario acordado con cliente: {horario}

Al final del mensaje, indica que responda "llegué" o "SI" cuando llegue al domicilio.
Formato: mensaje de WhatsApp, máximo 20 líneas, con secciones claras.
""".strip()

PROMPT_PREGUNTA_FALLA = """
El técnico acaba de indicar que terminó la visita (escribió: "{respuesta_tecnico}").
Inicia el proceso de recopilación de información para cerrar el ticket.

Envíale un mensaje confirmando que anotaste que terminó y hazle la PRIMERA pregunta:
¿Qué tipo de falla encontraste en el domicilio?

Da ejemplos entre paréntesis: fibra cortada, ONT dañada, conector sucio, problema de configuración, 
cable interno dañado, interferencia, etc.

Tono: directo y amigable. Máximo 3 líneas.
""".strip()

PROMPT_PREGUNTA_SOLUCION = """
El técnico respondió sobre el tipo de falla: "{respuesta_tecnico}"

Confirma que anotaste su respuesta con un ✅ y hazle la SEGUNDA pregunta:
¿Qué solución aplicaste para resolver el problema?

Da ejemplos: reemplazo de ONT, limpieza de conectores, empalme de fibra, reconfiguración, 
cambio de cable, etc.

Máximo 2 líneas.
""".strip()

PROMPT_PREGUNTA_MATERIALES = """
El técnico respondió sobre la solución: "{respuesta_tecnico}"

Confirma con ✅ y hazle la TERCERA pregunta:
¿Utilizaste materiales o repuestos? 

Si sí: que especifique cuáles y qué cantidad.
Si no: que responda "ninguno".

Máximo 2 líneas.
""".strip()

PROMPT_PREGUNTA_FOTO = """
El técnico respondió sobre materiales: "{respuesta_tecnico}"

Confirma con ✅ y hazle la CUARTA y última pregunta:
Por favor envía una foto de evidencia del trabajo realizado 
(equipo instalado, conexiones, o el estado final del sitio).

Menciona que es la última pregunta y que después cerrará el ticket automáticamente.
Máximo 2 líneas.
""".strip()

PROMPT_CIERRE_TECNICO = """
Recopilaste toda la información del técnico. El ticket ya fue cerrado en el sistema.
Envía un mensaje de confirmación al técnico con:

1. Confirmación de que el ticket fue cerrado exitosamente
2. Resumen de lo registrado:
   - Falla: {tipo_falla}
   - Solución: {solucion}
   - Materiales: {materiales}
   - Hora llegada: {hora_checkin}
   - Hora cierre: {hora_cierre}
   - Tiempo en sitio: {tiempo_sitio} minutos
3. Agradecimiento por el trabajo
4. Mención de que el cliente ya fue notificado

Tono: positivo, breve. Máximo 8 líneas.
""".strip()

PROMPT_CHECKIN = """
El técnico respondió confirmando que llegó al domicilio: "{respuesta_tecnico}"

Envía un mensaje confirmando:
1. Que registraste su llegada con hora exacta: {hora_checkin}
2. Que el cliente ya fue notificado de que está en camino
3. Que cuando termine, escriba "listo" para iniciar el cierre del ticket

Máximo 3 líneas. Tono motivador.
""".strip()


# ─────────────────────────────────────────────
# ESTADOS DE SESIÓN DEL TÉCNICO
# ─────────────────────────────────────────────

FASES_TECNICO = {
    "ESPERANDO_CHECKIN":     "Técnico recibió el brief, esperando confirmación de llegada",
    "EN_SITIO":              "Técnico hizo check-in, trabajando en el domicilio",
    "RECOPILANDO_FALLA":     "IA preguntando tipo de falla",
    "RECOPILANDO_SOLUCION":  "IA preguntando solución aplicada",
    "RECOPILANDO_MATERIALES":"IA preguntando materiales usados",
    "ESPERANDO_FOTO":        "IA esperando foto de evidencia",
    "CERRADO":               "Ticket cerrado, conversación finalizada",
}


# ─────────────────────────────────────────────
# KEYWORDS DE DETECCIÓN
# ─────────────────────────────────────────────

KEYWORDS_CHECKIN = [
    "llegué", "llegue", "llegué", "llegue", "si", "sí", "ok", "listo",
    "estoy aquí", "ya llegue", "aqui", "aquí", "confirmo", "llegamos"
]

KEYWORDS_TERMINADO = [
    "listo", "terminé", "termine", "ya", "listo ya", "finalicé",
    "finalice", "resuelto", "arreglé", "arregle", "solucioné",
    "solucionado", "completado", "hecho"
]


def es_checkin(texto: str) -> bool:
    return any(k in texto.lower() for k in KEYWORDS_CHECKIN)

def es_terminado(texto: str) -> bool:
    return any(k in texto.lower() for k in KEYWORDS_TERMINADO)

def es_foto(msg_type: str) -> bool:
    return msg_type in ("image", "document")


# ─────────────────────────────────────────────
# PROCESADOR PRINCIPAL DEL TÉCNICO
# ─────────────────────────────────────────────

async def procesar_mensaje_tecnico(
    phone_tecnico: str,
    mensaje: str,
    msg_type: str,
    media_id: Optional[str],
    bg: BackgroundTasks,
    # Inyectados desde main.py
    get_session_fn,
    save_session_fn,
    call_glm_fn,
    wa_send_fn,
    mw_update_ticket_fn,
    wa_send_cliente_fn,
    notificar_cliente_fn,
    descargar_media_fn,
    adjuntar_foto_ticket_fn,
):
    """
    Orquestador de la conversación con el técnico.
    Gestiona el estado paso a paso y actualiza MikroWisp
    en cada etapa sin intervención humana.
    """
    # Prefijo para diferenciar sesión de técnico vs cliente
    session = await get_session_fn(f"tec:{phone_tecnico}")

    fase = session.fase

    # ── FASE: ESPERANDO CHECK-IN ─────────────
    if fase == "ESPERANDO_CHECKIN":

        if es_checkin(mensaje):
            hora_checkin = datetime.now().strftime("%I:%M %p")
            session.fase = "EN_SITIO"

            # Guardar hora de check-in en sesión
            if not hasattr(session, 'extra'):
                session.extra = {}
            session.extra["hora_checkin"] = hora_checkin
            session.extra["hora_checkin_iso"] = datetime.now().isoformat()

            # 1. Actualizar ticket en MikroWisp con check-in
            await mw_update_ticket_fn(session.ticket_id, {
                "estado": "en_progreso",
                "hora_llegada_tecnico": datetime.now().isoformat(),
                "notas": f"Técnico llegó al domicilio a las {hora_checkin}"
            })

            # 2. Responder al técnico
            prompt = PROMPT_CHECKIN.format(
                respuesta_tecnico=mensaje,
                hora_checkin=hora_checkin
            )
            reply = await call_glm_fn(prompt, session)
            await wa_send_fn(phone_tecnico, reply)

            # 3. Notificar al cliente (conversación paralela)
            bg.add_task(
                notificar_cliente_fn,
                session.extra.get("phone_cliente"),
                "tecnico_en_camino"
            )

            await save_session_fn(session)

        else:
            # El técnico escribió algo que no es check-in
            await wa_send_fn(
                phone_tecnico,
                f"Hola! Cuando llegues al domicilio de *{session.nombre}*, "
                f"responde *SÍ* para registrar tu llegada y notificar al cliente. "
                f"Recuerda la dirección: {session.extra.get('direccion', 'Ver brief anterior')} 📍"
            )
        return

    # ── FASE: EN SITIO (trabajando) ───────────
    elif fase == "EN_SITIO":

        if es_terminado(mensaje):
            session.fase = "RECOPILANDO_FALLA"
            session.extra["hora_inicio_cierre"] = datetime.now().isoformat()

            prompt = PROMPT_PREGUNTA_FALLA.format(respuesta_tecnico=mensaje)
            reply = await call_glm_fn(prompt, session)
            await wa_send_fn(phone_tecnico, reply)
            await save_session_fn(session)
        else:
            # Mensaje durante el trabajo → recordatorio amable
            await wa_send_fn(
                phone_tecnico,
                "Entendido 👍 Cuando termines escríbeme *listo* para registrar el cierre del ticket."
            )
        return

    # ── FASE: RECOPILANDO TIPO DE FALLA ──────
    elif fase == "RECOPILANDO_FALLA":

        session.extra["tipo_falla"] = mensaje
        session.fase = "RECOPILANDO_SOLUCION"

        # Actualizar ticket parcialmente en MikroWisp
        await mw_update_ticket_fn(session.ticket_id, {
            "tipo_falla": mensaje
        })

        prompt = PROMPT_PREGUNTA_SOLUCION.format(respuesta_tecnico=mensaje)
        reply = await call_glm_fn(prompt, session)
        await wa_send_fn(phone_tecnico, reply)
        await save_session_fn(session)
        return

    # ── FASE: RECOPILANDO SOLUCIÓN ────────────
    elif fase == "RECOPILANDO_SOLUCION":

        session.extra["solucion"] = mensaje
        session.fase = "RECOPILANDO_MATERIALES"

        await mw_update_ticket_fn(session.ticket_id, {
            "solucion_aplicada": mensaje
        })

        prompt = PROMPT_PREGUNTA_MATERIALES.format(respuesta_tecnico=mensaje)
        reply = await call_glm_fn(prompt, session)
        await wa_send_fn(phone_tecnico, reply)
        await save_session_fn(session)
        return

    # ── FASE: RECOPILANDO MATERIALES ──────────
    elif fase == "RECOPILANDO_MATERIALES":

        session.extra["materiales"] = mensaje
        session.fase = "ESPERANDO_FOTO"

        await mw_update_ticket_fn(session.ticket_id, {
            "materiales_usados": mensaje
        })

        prompt = PROMPT_PREGUNTA_FOTO.format(respuesta_tecnico=mensaje)
        reply = await call_glm_fn(prompt, session)
        await wa_send_fn(phone_tecnico, reply)
        await save_session_fn(session)
        return

    # ── FASE: ESPERANDO FOTO ──────────────────
    elif fase == "ESPERANDO_FOTO":

        if es_foto(msg_type) and media_id:
            # Descargar y adjuntar foto al ticket
            foto_url = await descargar_media_fn(media_id)
            if foto_url:
                await adjuntar_foto_ticket_fn(session.ticket_id, foto_url)

            # Calcular tiempo en sitio
            hora_checkin_iso = session.extra.get("hora_checkin_iso")
            hora_cierre = datetime.now()
            tiempo_sitio = 0
            if hora_checkin_iso:
                checkin_dt = datetime.fromisoformat(hora_checkin_iso)
                tiempo_sitio = int((hora_cierre - checkin_dt).total_seconds() / 60)

            hora_cierre_str = hora_cierre.strftime("%I:%M %p")

            # Cerrar ticket completo en MikroWisp
            await mw_update_ticket_fn(session.ticket_id, {
                "estado": "cerrado",
                "tipo_falla": session.extra.get("tipo_falla"),
                "solucion_aplicada": session.extra.get("solucion"),
                "materiales_usados": session.extra.get("materiales"),
                "hora_llegada_tecnico": session.extra.get("hora_checkin_iso"),
                "hora_cierre": hora_cierre.isoformat(),
                "tiempo_en_sitio_min": tiempo_sitio,
                "foto_evidencia_url": foto_url,
                "cerrado_por": "ia_automatico",
            })

            # Responder al técnico con resumen
            prompt = PROMPT_CIERRE_TECNICO.format(
                tipo_falla=session.extra.get("tipo_falla", "N/A"),
                solucion=session.extra.get("solucion", "N/A"),
                materiales=session.extra.get("materiales", "Ninguno"),
                hora_checkin=session.extra.get("hora_checkin", "N/A"),
                hora_cierre=hora_cierre_str,
                tiempo_sitio=tiempo_sitio,
            )
            reply = await call_glm_fn(prompt, session)
            await wa_send_fn(phone_tecnico, reply)

            # Notificar al cliente que el servicio fue restaurado + CSAT
            bg.add_task(
                notificar_cliente_fn,
                session.extra.get("phone_cliente"),
                "servicio_restaurado"
            )

            session.fase = "CERRADO"
            await save_session_fn(session)

        else:
            # No envió foto → recordatorio
            await wa_send_fn(
                phone_tecnico,
                "Necesito que envíes una foto 📷 del trabajo realizado para poder cerrar el ticket. "
                "Puede ser del equipo instalado o las conexiones finales."
            )
        return

    # ── FASE: CERRADO ─────────────────────────
    elif fase == "CERRADO":
        await wa_send_fn(
            phone_tecnico,
            "Este ticket ya está cerrado ✅. Si tienes un nuevo caso, el sistema te lo enviará automáticamente."
        )
        return


# ─────────────────────────────────────────────
# FUNCIÓN: NOTIFICAR AL CLIENTE
# Se llama desde background tasks
# ─────────────────────────────────────────────

MENSAJES_CLIENTE = {
    "ticket_creado": (
        "📋 Tu reporte fue registrado con el número *#{ticket_id}*.\n"
        "Un técnico fue asignado a tu caso y te contactaremos para confirmar el horario de visita. "
        "Puedes escribirme si tienes alguna pregunta 🙏"
    ),
    "tecnico_asignado": (
        "🔧 Tu técnico ya fue asignado al ticket *#{ticket_id}*.\n"
        "Te avisaré cuando esté en camino hacia tu domicilio."
    ),
    "tecnico_en_camino": (
        "🚗 ¡Buenas noticias! Tu técnico ya está en camino.\n"
        "Llegará en breve a tu domicilio. Por favor asegúrate de estar disponible para recibirlo."
    ),
    "servicio_restaurado": (
        "✅ ¡Tu servicio de internet ha sido restaurado exitosamente!\n\n"
        "¿Podrías calificarnos del *1 al 5*? Tu opinión nos ayuda a mejorar:\n"
        "1️⃣ Muy malo  2️⃣ Malo  3️⃣ Regular  4️⃣ Bueno  5️⃣ Excelente"
    ),
}

async def construir_mensaje_cliente(evento: str, datos: dict = {}) -> str:
    """Construye el mensaje para el cliente según el evento"""
    template = MENSAJES_CLIENTE.get(evento, "Tu caso está siendo atendido. 🙏")
    try:
        return template.format(**datos)
    except KeyError:
        return template


# ─────────────────────────────────────────────
# FUNCIÓN: PREPARAR SESIÓN DEL TÉCNICO
# Se llama cuando se crea un nuevo ticket
# ─────────────────────────────────────────────

def preparar_sesion_tecnico(
    phone_tecnico: str,
    ticket_id: str,
    datos_cliente: dict,
    datos_ont: dict,
    phone_cliente: str
) -> dict:
    """
    Prepara el estado inicial de la sesión del técnico
    cuando se le asigna un nuevo ticket.
    """
    return {
        "phone": f"tec:{phone_tecnico}",
        "fase": "ESPERANDO_CHECKIN",
        "ticket_id": ticket_id,
        "nombre": datos_cliente.get("nombre"),
        "contrato": datos_cliente.get("contrato"),
        "historial": [],
        "extra": {
            "phone_cliente": phone_cliente,
            "nombre_cliente": datos_cliente.get("nombre"),
            "direccion": datos_cliente.get("direccion"),
            "telefono_cliente": datos_cliente.get("telefono"),
            "plan": datos_cliente.get("plan"),
            "serial_ont": datos_ont.get("serial"),
            "señal_dbm": datos_ont.get("señal"),
            "estado_ont": datos_ont.get("estado"),
            "hora_asignacion": datetime.now().isoformat(),
            "tipo_falla": None,
            "solucion": None,
            "materiales": None,
            "hora_checkin": None,
            "hora_checkin_iso": None,
        },
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
