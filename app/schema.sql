-- =============================================================
--  ISP AI SUPPORT SYSTEM — ESQUEMA DE BASE DE DATOS
--  PostgreSQL 15
-- =============================================================
--  Archivo: schema.sql
--  Descripción: Estructura completa de la base de datos
--               para el sistema de soporte IA
-- =============================================================

-- Extensiones
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- Para búsqueda por texto

-- ─────────────────────────────────────────────────────────────
-- TABLA: conversaciones
-- Registro de cada conversación de soporte iniciada
-- ─────────────────────────────────────────────────────────────
CREATE TABLE conversaciones (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone           VARCHAR(20)  NOT NULL,           -- Número WhatsApp del cliente
    contrato        VARCHAR(50),                      -- Contrato en MikroWisp
    nombre_cliente  VARCHAR(150),
    fase_final      VARCHAR(50),                      -- Fase en que terminó
    resultado       VARCHAR(30),                      -- RESUELTO_IA / VISITA_TECNICA / MORA / MASIVO
    ticket_id       VARCHAR(50),                      -- ID del ticket en MikroWisp
    serial_ont      VARCHAR(100),                     -- Serial de la ONT (SmartOLT)
    reboot_ejecutado BOOLEAN DEFAULT FALSE,
    señal_inicial   DECIMAL(6,2),                    -- dBm al inicio
    señal_final     DECIMAL(6,2),                    -- dBm al cierre
    iniciada_en     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cerrada_en      TIMESTAMPTZ,
    -- KPIs calculados al cierre
    frt_segundos    INTEGER,     -- First Response Time
    ttr_minutos     INTEGER,     -- Time to Resolution
    fcr             BOOLEAN,     -- First Contact Resolution (sin visita técnica)
    csat            SMALLINT CHECK (csat BETWEEN 1 AND 5)
);

CREATE INDEX idx_conv_phone     ON conversaciones(phone);
CREATE INDEX idx_conv_contrato  ON conversaciones(contrato);
CREATE INDEX idx_conv_resultado ON conversaciones(resultado);
CREATE INDEX idx_conv_fecha     ON conversaciones(iniciada_en DESC);


-- ─────────────────────────────────────────────────────────────
-- TABLA: mensajes
-- Historial completo de mensajes de cada conversación
-- ─────────────────────────────────────────────────────────────
CREATE TABLE mensajes (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversacion_id  UUID NOT NULL REFERENCES conversaciones(id) ON DELETE CASCADE,
    rol              VARCHAR(15) NOT NULL CHECK (rol IN ('cliente', 'ia', 'sistema')),
    contenido        TEXT NOT NULL,
    fase             VARCHAR(50),                     -- Fase del flujo al momento del mensaje
    tokens_usados    INTEGER,                         -- Tokens consumidos en la llamada GLM
    latencia_ms      INTEGER,                         -- Tiempo de respuesta del LLM
    creado_en        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_msg_conversacion ON mensajes(conversacion_id);
CREATE INDEX idx_msg_fecha        ON mensajes(creado_en DESC);


-- ─────────────────────────────────────────────────────────────
-- TABLA: acciones_ia
-- Log de acciones autónomas ejecutadas por el IA
-- (consultas a APIs, reinicios, creación de tickets, etc.)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE acciones_ia (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversacion_id  UUID NOT NULL REFERENCES conversaciones(id) ON DELETE CASCADE,
    tipo_accion      VARCHAR(50) NOT NULL,
    -- Tipos: CONSULTA_MIKROWISP, CONSULTA_SMARTOLT, REBOOT_ONT,
    --        CREAR_TICKET, CERRAR_TICKET, ENVIAR_CSAT, NOTIF_TECNICO
    sistema          VARCHAR(20),                     -- MIKROWISP / SMARTOLT / WHATSAPP / GLM
    endpoint         VARCHAR(200),                    -- Endpoint llamado
    metodo_http      VARCHAR(10),                     -- GET / POST / PUT
    payload          JSONB,                           -- Datos enviados
    respuesta        JSONB,                           -- Respuesta recibida
    exitoso          BOOLEAN NOT NULL DEFAULT TRUE,
    error_msg        TEXT,
    latencia_ms      INTEGER,
    ejecutado_en     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_accion_conv   ON acciones_ia(conversacion_id);
CREATE INDEX idx_accion_tipo   ON acciones_ia(tipo_accion);
CREATE INDEX idx_accion_fecha  ON acciones_ia(ejecutado_en DESC);
CREATE INDEX idx_accion_error  ON acciones_ia(exitoso) WHERE exitoso = FALSE;


-- ─────────────────────────────────────────────────────────────
-- TABLA: kpis_diarios
-- Resumen diario de KPIs calculados automáticamente
-- Se actualiza por Celery cada medianoche
-- ─────────────────────────────────────────────────────────────
CREATE TABLE kpis_diarios (
    id                     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fecha                  DATE NOT NULL UNIQUE,
    -- Volumen
    total_conversaciones   INTEGER DEFAULT 0,
    total_tickets_creados  INTEGER DEFAULT 0,
    -- FRT — First Response Time
    frt_promedio_seg       DECIMAL(8,2),
    frt_min_seg            INTEGER,
    frt_max_seg            INTEGER,
    -- FCR — First Contact Resolution
    fcr_total              INTEGER DEFAULT 0,        -- Casos resueltos sin técnico
    fcr_porcentaje         DECIMAL(5,2),             -- % FCR
    -- TTR — Time to Resolution
    ttr_promedio_min       DECIMAL(8,2),
    ttr_tier1_promedio_min DECIMAL(8,2),             -- Solo IA
    ttr_tier2_promedio_min DECIMAL(8,2),             -- Con visita técnica
    -- CSAT
    csat_promedio          DECIMAL(3,2),
    csat_total_respuestas  INTEGER DEFAULT 0,
    csat_5_estrellas       INTEGER DEFAULT 0,
    csat_1_estrella        INTEGER DEFAULT 0,
    -- Por tipo de resolución
    resueltos_remotamente  INTEGER DEFAULT 0,
    resueltos_visita       INTEGER DEFAULT 0,
    casos_mora             INTEGER DEFAULT 0,
    casos_corte_masivo     INTEGER DEFAULT 0,
    -- Infraestructura
    reboots_exitosos       INTEGER DEFAULT 0,
    reboots_fallidos       INTEGER DEFAULT 0,
    -- GLM
    total_tokens_glm       INTEGER DEFAULT 0,
    costo_estimado_usd     DECIMAL(8,4),
    calculado_en           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_kpis_fecha ON kpis_diarios(fecha DESC);


-- ─────────────────────────────────────────────────────────────
-- TABLA: alertas_sla
-- Alertas cuando un KPI supera el umbral definido
-- ─────────────────────────────────────────────────────────────
CREATE TABLE alertas_sla (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tipo             VARCHAR(50) NOT NULL,
    -- Tipos: FRT_EXCEDIDO, TTR_EXCEDIDO, CSAT_BAJO, FCR_BAJO
    conversacion_id  UUID REFERENCES conversaciones(id),
    descripcion      TEXT NOT NULL,
    valor_actual     DECIMAL(10,2),
    umbral           DECIMAL(10,2),
    resuelta         BOOLEAN DEFAULT FALSE,
    creada_en        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resuelta_en      TIMESTAMPTZ
);

CREATE INDEX idx_alerta_tipo     ON alertas_sla(tipo);
CREATE INDEX idx_alerta_resuelta ON alertas_sla(resuelta) WHERE resuelta = FALSE;


-- ─────────────────────────────────────────────────────────────
-- TABLA: fallas_ont
-- Historial de fallas y reboots por ONT (SmartOLT)
-- Útil para detectar equipos con fallas recurrentes
-- ─────────────────────────────────────────────────────────────
CREATE TABLE fallas_ont (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    serial_ont       VARCHAR(100) NOT NULL,
    contrato         VARCHAR(50),
    tipo_falla       VARCHAR(50),                     -- OFFLINE / SEÑAL_BAJA / INTERMITENCIA
    señal_dbm        DECIMAL(6,2),
    reboot_exitoso   BOOLEAN,
    requirio_visita  BOOLEAN DEFAULT FALSE,
    registrada_en    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_falla_serial ON fallas_ont(serial_ont);
CREATE INDEX idx_falla_fecha  ON fallas_ont(registrada_en DESC);

-- Vista para detectar ONTs con fallas recurrentes (>3 en 30 días)
CREATE VIEW onts_problematicas AS
SELECT
    serial_ont,
    contrato,
    COUNT(*) AS total_fallas,
    SUM(CASE WHEN reboot_exitoso THEN 1 ELSE 0 END) AS reboots_exitosos,
    SUM(CASE WHEN requirio_visita THEN 1 ELSE 0 END) AS visitas_requeridas,
    AVG(señal_dbm) AS señal_promedio,
    MAX(registrada_en) AS ultima_falla
FROM fallas_ont
WHERE registrada_en >= NOW() - INTERVAL '30 days'
GROUP BY serial_ont, contrato
HAVING COUNT(*) >= 3
ORDER BY total_fallas DESC;


-- ─────────────────────────────────────────────────────────────
-- TABLA: config_isp
-- Configuración del ISP almacenada en DB (editable sin redeploy)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE config_isp (
    clave   VARCHAR(100) PRIMARY KEY,
    valor   TEXT NOT NULL,
    tipo    VARCHAR(20) DEFAULT 'string',  -- string / integer / boolean / json
    descripcion TEXT,
    actualizado_en TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO config_isp (clave, valor, tipo, descripcion) VALUES
('isp_nombre',              'Tu ISP',                     'string',  'Nombre del ISP'),
('horario_tecnico',         'Lunes-Sábado 8am-6pm',       'string',  'Horario de técnicos de campo'),
('numero_pagos',            '+57 300 000 0000',           'string',  'Número para pagos/admin'),
('señal_minima_dbm',        '-27',                        'integer', 'Umbral mínimo señal óptica'),
('señal_maxima_dbm',        '-8',                         'integer', 'Umbral máximo señal óptica'),
('reboot_wait_seg',         '120',                        'integer', 'Segundos espera post-reboot'),
('sla_frt_seg',             '30',                        'integer', 'SLA: FRT máximo en segundos'),
('sla_ttr_min_tier1',       '10',                        'integer', 'SLA: TTR máximo Tier1 (min)'),
('sla_ttr_min_tier2',       '240',                       'integer', 'SLA: TTR máximo Tier2 (min)'),
('sla_csat_minimo',         '4.0',                       'string',  'CSAT mínimo aceptable'),
('sla_fcr_porcentaje',      '60',                        'integer', 'FCR objetivo en porcentaje'),
('alarma_clientes_masivo',  '3',                         'integer', 'Mín. clientes para corte masivo');


-- ─────────────────────────────────────────────────────────────
-- VISTA: dashboard_tiempo_real
-- Métricas del día actual para el dashboard
-- ─────────────────────────────────────────────────────────────
CREATE VIEW dashboard_tiempo_real AS
SELECT
    COUNT(*) FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE)        AS conversaciones_hoy,
    COUNT(*) FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE AND fcr) AS fcr_hoy,
    ROUND(AVG(frt_segundos) FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE), 1) AS frt_promedio_hoy,
    ROUND(AVG(ttr_minutos)  FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE), 1) AS ttr_promedio_hoy,
    ROUND(AVG(csat)         FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE AND csat IS NOT NULL), 2) AS csat_promedio_hoy,
    COUNT(*) FILTER (WHERE fase_final = 'ESPERANDO_TECNICO')         AS tickets_abiertos,
    COUNT(*) FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE AND resultado = 'RESUELTO_IA') AS resueltos_remotamente_hoy
FROM conversaciones;
