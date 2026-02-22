-- =============================================================
--  ISP AI SUPPORT SYSTEM — ESQUEMA DE BASE DE DATOS
--  PostgreSQL 15 — Compatible con Docker Alpine
--  v3.1 — Corregido para EasyPanel
-- =============================================================

-- Extension UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────────────────────
-- TABLA: conversaciones
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversaciones (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone             VARCHAR(20)  NOT NULL,
    contrato          VARCHAR(50),
    nombre_cliente    VARCHAR(150),
    fase_final        VARCHAR(50),
    resultado         VARCHAR(30),
    ticket_id         VARCHAR(50),
    serial_ont        VARCHAR(100),
    reboot_ejecutado  BOOLEAN DEFAULT FALSE,
    senal_inicial     DECIMAL(6,2),
    senal_final       DECIMAL(6,2),
    iniciada_en       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cerrada_en        TIMESTAMPTZ,
    frt_segundos      INTEGER,
    ttr_minutos       INTEGER,
    fcr               BOOLEAN,
    csat              SMALLINT CHECK (csat BETWEEN 1 AND 5)
);

CREATE INDEX IF NOT EXISTS idx_conv_phone     ON conversaciones(phone);
CREATE INDEX IF NOT EXISTS idx_conv_contrato  ON conversaciones(contrato);
CREATE INDEX IF NOT EXISTS idx_conv_resultado ON conversaciones(resultado);
CREATE INDEX IF NOT EXISTS idx_conv_fecha     ON conversaciones(iniciada_en DESC);


-- ─────────────────────────────────────────────────────────────
-- TABLA: mensajes
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS mensajes (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversacion_id  UUID NOT NULL REFERENCES conversaciones(id) ON DELETE CASCADE,
    rol              VARCHAR(15) NOT NULL CHECK (rol IN ('cliente', 'ia', 'sistema')),
    contenido        TEXT NOT NULL,
    fase             VARCHAR(50),
    tokens_usados    INTEGER,
    latencia_ms      INTEGER,
    creado_en        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_msg_conversacion ON mensajes(conversacion_id);
CREATE INDEX IF NOT EXISTS idx_msg_fecha        ON mensajes(creado_en DESC);


-- ─────────────────────────────────────────────────────────────
-- TABLA: acciones_ia
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS acciones_ia (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversacion_id  UUID NOT NULL REFERENCES conversaciones(id) ON DELETE CASCADE,
    tipo_accion      VARCHAR(50) NOT NULL,
    sistema          VARCHAR(20),
    endpoint         VARCHAR(200),
    metodo_http      VARCHAR(10),
    payload          JSONB,
    respuesta        JSONB,
    exitoso          BOOLEAN NOT NULL DEFAULT TRUE,
    error_msg        TEXT,
    latencia_ms      INTEGER,
    ejecutado_en     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_accion_conv   ON acciones_ia(conversacion_id);
CREATE INDEX IF NOT EXISTS idx_accion_tipo   ON acciones_ia(tipo_accion);
CREATE INDEX IF NOT EXISTS idx_accion_fecha  ON acciones_ia(ejecutado_en DESC);


-- ─────────────────────────────────────────────────────────────
-- TABLA: kpis_diarios
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kpis_diarios (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fecha                   DATE NOT NULL UNIQUE,
    total_conversaciones    INTEGER DEFAULT 0,
    total_tickets_creados   INTEGER DEFAULT 0,
    frt_promedio_seg        DECIMAL(8,2),
    frt_min_seg             INTEGER,
    frt_max_seg             INTEGER,
    fcr_total               INTEGER DEFAULT 0,
    fcr_porcentaje          DECIMAL(5,2),
    ttr_promedio_min        DECIMAL(8,2),
    ttr_tier1_promedio_min  DECIMAL(8,2),
    ttr_tier2_promedio_min  DECIMAL(8,2),
    csat_promedio           DECIMAL(3,2),
    csat_total_respuestas   INTEGER DEFAULT 0,
    csat_5_estrellas        INTEGER DEFAULT 0,
    csat_1_estrella         INTEGER DEFAULT 0,
    resueltos_remotamente   INTEGER DEFAULT 0,
    resueltos_visita        INTEGER DEFAULT 0,
    casos_mora              INTEGER DEFAULT 0,
    casos_corte_masivo      INTEGER DEFAULT 0,
    reboots_exitosos        INTEGER DEFAULT 0,
    reboots_fallidos        INTEGER DEFAULT 0,
    total_tokens_glm        INTEGER DEFAULT 0,
    costo_estimado_usd      DECIMAL(8,4),
    calculado_en            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kpis_fecha ON kpis_diarios(fecha DESC);


-- ─────────────────────────────────────────────────────────────
-- TABLA: alertas_sla
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alertas_sla (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tipo             VARCHAR(50) NOT NULL,
    conversacion_id  UUID REFERENCES conversaciones(id),
    descripcion      TEXT NOT NULL,
    valor_actual     DECIMAL(10,2),
    umbral           DECIMAL(10,2),
    resuelta         BOOLEAN DEFAULT FALSE,
    creada_en        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resuelta_en      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_alerta_tipo     ON alertas_sla(tipo);
CREATE INDEX IF NOT EXISTS idx_alerta_resuelta ON alertas_sla(resuelta) WHERE resuelta = FALSE;


-- ─────────────────────────────────────────────────────────────
-- TABLA: fallas_ont
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fallas_ont (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    serial_ont       VARCHAR(100) NOT NULL,
    contrato         VARCHAR(50),
    tipo_falla       VARCHAR(50),
    senal_dbm        DECIMAL(6,2),
    reboot_exitoso   BOOLEAN,
    requirio_visita  BOOLEAN DEFAULT FALSE,
    registrada_en    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_falla_serial ON fallas_ont(serial_ont);
CREATE INDEX IF NOT EXISTS idx_falla_fecha  ON fallas_ont(registrada_en DESC);


-- ─────────────────────────────────────────────────────────────
-- TABLA: config_isp
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS config_isp (
    clave          VARCHAR(100) PRIMARY KEY,
    valor          TEXT NOT NULL,
    tipo           VARCHAR(20) DEFAULT 'string',
    descripcion    TEXT,
    actualizado_en TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO config_isp (clave, valor, tipo, descripcion) VALUES
('isp_nombre',             'Interlan',                   'string',  'Nombre del ISP'),
('horario_tecnico',        'Lunes-Sabado 8am-6pm',       'string',  'Horario de tecnicos de campo'),
('numero_pagos',           '+51 916 902 397',            'string',  'Numero para pagos/admin'),
('senal_minima_dbm',       '-27',                        'integer', 'Umbral minimo senal optica'),
('senal_maxima_dbm',       '-8',                         'integer', 'Umbral maximo senal optica'),
('reboot_wait_seg',        '120',                        'integer', 'Segundos espera post-reboot'),
('sla_frt_seg',            '30',                         'integer', 'SLA: FRT maximo en segundos'),
('sla_ttr_min_tier1',      '10',                         'integer', 'SLA: TTR maximo Tier1 min'),
('sla_ttr_min_tier2',      '240',                        'integer', 'SLA: TTR maximo Tier2 min'),
('sla_csat_minimo',        '4.0',                        'string',  'CSAT minimo aceptable'),
('sla_fcr_porcentaje',     '60',                         'integer', 'FCR objetivo en porcentaje'),
('alarma_clientes_masivo', '3',                          'integer', 'Min clientes para corte masivo')
ON CONFLICT (clave) DO NOTHING;


-- ─────────────────────────────────────────────────────────────
-- VISTA: dashboard_tiempo_real
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW dashboard_tiempo_real AS
SELECT
    COUNT(*) FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE)                          AS conversaciones_hoy,
    COUNT(*) FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE AND fcr = TRUE)           AS fcr_hoy,
    ROUND(AVG(frt_segundos) FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE), 1)       AS frt_promedio_hoy,
    ROUND(AVG(ttr_minutos)  FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE), 1)       AS ttr_promedio_hoy,
    ROUND(AVG(csat)         FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE
                                    AND csat IS NOT NULL), 2)                          AS csat_promedio_hoy,
    COUNT(*) FILTER (WHERE fase_final = 'ESPERANDO_TECNICO')                          AS tickets_abiertos,
    COUNT(*) FILTER (WHERE DATE(iniciada_en) = CURRENT_DATE
                     AND resultado = 'RESUELTO_IA')                                    AS resueltos_remotamente_hoy
FROM conversaciones;
