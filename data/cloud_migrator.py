#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║               CLOUD MIGRATOR — SQLite → Supabase                ║
║          Migración de nba_historical.db a PostgreSQL            ║
╚══════════════════════════════════════════════════════════════════╝

Requisitos:
    pip install supabase tqdm python-dotenv

Uso:
    1. Crea un archivo .env en la misma carpeta con:
         SUPABASE_URL=https://gvifnndzhsczgbwmxyti.supabase.co
         SUPABASE_KEY=tu_service_role_key_aqui
    2. Ejecuta:  python cloud_migrator.py
"""

import os
import sys
import sqlite3
import json
import time
import math
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("❌ Falta 'tqdm'. Instálala con: pip install tqdm")

try:
    from supabase import create_client, Client
except ImportError:
    sys.exit("❌ Falta 'supabase'. Instálala con: pip install supabase")

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("❌ Falta 'python-dotenv'. Instálala con: pip install python-dotenv")


# ─────────────────────────── CONFIGURACIÓN ────────────────────────────

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://gvifnndzhsczgbwmxyti.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "nba_historical.db")
BATCH_SIZE = 500          # Filas por lote (evita timeouts)
RETRY_ATTEMPTS = 3        # Reintentos por lote fallido
RETRY_DELAY = 2           # Segundos entre reintentos
LOG_FILE = "migration_log.txt"


# ─────────────────────── MAPEO DE TIPOS ───────────────────────────────

SQLITE_TO_POSTGRES = {
    "INTEGER":  "BIGINT",
    "REAL":     "DOUBLE PRECISION",
    "TEXT":     "TEXT",
    "BLOB":     "BYTEA",
    "NUMERIC":  "NUMERIC",
    "BOOLEAN":  "BOOLEAN",
    "DATE":     "DATE",
    "DATETIME": "TIMESTAMP",
    "FLOAT":    "DOUBLE PRECISION",
    "DOUBLE":   "DOUBLE PRECISION",
    "VARCHAR":  "TEXT",
    "CHAR":     "TEXT",
    "INT":      "BIGINT",
    "BIGINT":   "BIGINT",
    "SMALLINT": "SMALLINT",
}


# ─────────────────────── UTILIDADES ───────────────────────────────────

class MigrationLogger:
    """Logger dual: consola + archivo."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"═══ Migración iniciada: {datetime.now().isoformat()} ═══\n\n")

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERROR": "❌", "SKIP": "⏭️"}
        icon = icons.get(level, "•")
        line = f"[{timestamp}] {icon}  {message}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def summary(self, results: dict):
        sep = "═" * 55
        lines = [
            f"\n{sep}",
            "         RESUMEN DE MIGRACIÓN",
            sep,
            f"  Tablas procesadas:  {results['processed']}",
            f"  Tablas exitosas:    {results['success']}",
            f"  Tablas saltadas:    {results['skipped']}",
            f"  Tablas con error:   {results['errors']}",
            f"  Filas migradas:     {results['total_rows']:,}",
            f"  Tiempo total:       {results['elapsed']}",
            sep,
        ]
        for line in lines:
            print(line)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def map_sqlite_type(sqlite_type: str) -> str:
    """Convierte un tipo SQLite a su equivalente en PostgreSQL."""
    if not sqlite_type:
        return "TEXT"
    base = sqlite_type.upper().split("(")[0].strip()
    return SQLITE_TO_POSTGRES.get(base, "TEXT")


def sanitize_value(value):
    """Limpia valores problemáticos para la API de Supabase."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def get_user_choice(table_name: str) -> str:
    """Pregunta al usuario qué hacer con una tabla que ya existe."""
    while True:
        print(f"\n⚠️  La tabla '{table_name}' ya tiene datos en Supabase.")
        print("    [s] Saltarla  |  [o] Sobrescribir (borrar datos y resubir)  |  [a] Agregar (append)")
        choice = input("    Tu elección: ").strip().lower()
        if choice in ("s", "o", "a"):
            return choice
        print("    Opción no válida. Intenta de nuevo.")


# ─────────────────────── LECTURA SQLite ───────────────────────────────

def get_sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    """Obtiene la lista de tablas del archivo SQLite."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [row[0] for row in cursor.fetchall()]


def get_table_schema(conn: sqlite3.Connection, table: str) -> list[dict]:
    """Obtiene el esquema de una tabla (nombre, tipo, nullable, pk)."""
    cursor = conn.execute(f"PRAGMA table_info('{table}')")
    columns = []
    for row in cursor.fetchall():
        columns.append({
            "cid":      row[0],
            "name":     row[1],
            "type":     row[2],
            "notnull":  bool(row[3]),
            "default":  row[4],
            "pk":       bool(row[5]),
        })
    return columns


def get_row_count(conn: sqlite3.Connection, table: str) -> int:
    cursor = conn.execute(f"SELECT COUNT(*) FROM '{table}'")
    return cursor.fetchone()[0]


def read_rows_in_batches(conn: sqlite3.Connection, table: str, columns: list[str], batch_size: int):
    """Generador que entrega filas como dicts en lotes."""
    cursor = conn.execute(f"SELECT * FROM '{table}'")
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        batch = []
        for row in rows:
            record = {}
            for i, col in enumerate(columns):
                record[col] = sanitize_value(row[i])
            batch.append(record)
        yield batch


# ─────────────────────── OPERACIONES SUPABASE ─────────────────────────

def table_exists_in_supabase(supabase: Client, table: str) -> bool:
    """Verifica si una tabla ya tiene datos en Supabase."""
    try:
        response = supabase.table(table).select("*", count="exact").limit(1).execute()
        return response.count is not None and response.count > 0
    except Exception:
        return False


def create_table_via_sql(supabase: Client, table: str, columns: list[dict], logger: MigrationLogger):
    """
    Crea la tabla en Supabase usando la función RPC 'exec_sql'.

    IMPORTANTE: Debes crear esta función en tu Supabase SQL Editor:
    ──────────────────────────────────────────────────────────
    CREATE OR REPLACE FUNCTION exec_sql(query TEXT)
    RETURNS VOID AS $$
    BEGIN
      EXECUTE query;
    END;
    $$ LANGUAGE plpgsql SECURITY DEFINER;
    ──────────────────────────────────────────────────────────
    """
    col_defs = []
    pk_cols = [c["name"] for c in columns if c["pk"]]

    for col in columns:
        pg_type = map_sqlite_type(col["type"])
        parts = [f'"{col["name"]}"', pg_type]
        if col["notnull"]:
            parts.append("NOT NULL")
        if col["default"] is not None:
            parts.append(f"DEFAULT {col['default']}")
        col_defs.append(" ".join(parts))

    if pk_cols:
        pk_str = ", ".join(f'"{c}"' for c in pk_cols)
        col_defs.append(f"PRIMARY KEY ({pk_str})")

    ddl = f'CREATE TABLE IF NOT EXISTS "{table}" (\n  ' + ",\n  ".join(col_defs) + "\n);"
    logger.log(f"DDL para '{table}':\n{ddl}", "INFO")

    try:
        supabase.rpc("exec_sql", {"query": ddl}).execute()
        logger.log(f"Tabla '{table}' creada en Supabase.", "OK")
        return True
    except Exception as e:
        logger.log(f"Error creando tabla '{table}': {e}", "ERROR")
        logger.log("¿Creaste la función exec_sql? Revisa las instrucciones del script.", "WARN")
        return False


def truncate_table(supabase: Client, table: str, logger: MigrationLogger):
    """Vacía una tabla existente en Supabase."""
    try:
        supabase.rpc("exec_sql", {"query": f'TRUNCATE TABLE "{table}" CASCADE;'}).execute()
        logger.log(f"Tabla '{table}' vaciada.", "OK")
    except Exception as e:
        logger.log(f"Error vaciando '{table}': {e}", "ERROR")


def upload_batch(supabase: Client, table: str, batch: list[dict], logger: MigrationLogger) -> int:
    """Sube un lote de filas con reintentos."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            supabase.table(table).insert(batch).execute()
            return len(batch)
        except Exception as e:
            if attempt < RETRY_ATTEMPTS:
                logger.log(f"Reintento {attempt}/{RETRY_ATTEMPTS} en '{table}': {e}", "WARN")
                time.sleep(RETRY_DELAY * attempt)
            else:
                logger.log(f"Lote fallido en '{table}' tras {RETRY_ATTEMPTS} intentos: {e}", "ERROR")
                return 0
    return 0


# ─────────────────────── FLUJO PRINCIPAL ──────────────────────────────

def migrate():
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        🏀 NBA Historical DB → Supabase Cloud        ║
    ║              Migración SQLite → PostgreSQL           ║
    ╚══════════════════════════════════════════════════════╝
    """)

    # ── Validaciones iniciales ──
    if not SUPABASE_KEY:
        sys.exit("❌ SUPABASE_KEY no configurada. Agrégala en tu archivo .env")

    db_path = Path(SQLITE_DB_PATH)
    if not db_path.exists():
        sys.exit(f"❌ No se encontró la base de datos: {db_path.absolute()}")

    logger = MigrationLogger(LOG_FILE)
    logger.log(f"Base de datos: {db_path.absolute()} ({db_path.stat().st_size / 1024 / 1024:.1f} MB)")
    logger.log(f"Destino: {SUPABASE_URL}")

    # ── Conexiones ──
    sqlite_conn = sqlite3.connect(str(db_path))
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.log("Conexión a SQLite y Supabase establecida.", "OK")

    # ── Descubrimiento de tablas ──
    tables = get_sqlite_tables(sqlite_conn)
    if not tables:
        sys.exit("❌ No se encontraron tablas en el archivo SQLite.")

    logger.log(f"Tablas encontradas ({len(tables)}): {', '.join(tables)}")

    # ── Resumen previo ──
    print("\n┌─────────────────────────────────────────────┐")
    print("│  TABLA                         │    FILAS    │")
    print("├─────────────────────────────────────────────┤")
    total_rows_all = 0
    table_info = {}
    for t in tables:
        schema = get_table_schema(sqlite_conn, t)
        count = get_row_count(sqlite_conn, t)
        total_rows_all += count
        table_info[t] = {"schema": schema, "count": count}
        print(f"│  {t:<30s}│ {count:>10,} │")
    print("├─────────────────────────────────────────────┤")
    print(f"│  {'TOTAL':<30s}│ {total_rows_all:>10,} │")
    print("└─────────────────────────────────────────────┘")

    confirm = input("\n¿Iniciar migración? [s/N]: ").strip().lower()
    if confirm != "s":
        print("Migración cancelada.")
        sqlite_conn.close()
        return

    # ── Migración tabla por tabla ──
    results = {"processed": 0, "success": 0, "skipped": 0, "errors": 0, "total_rows": 0}
    start_time = time.time()

    for table in tables:
        results["processed"] += 1
        schema = table_info[table]["schema"]
        row_count = table_info[table]["count"]
        col_names = [c["name"] for c in schema]

        print(f"\n{'─' * 55}")
        logger.log(f"Procesando: {table} ({row_count:,} filas, {len(schema)} columnas)")

        # ── Verificar si existe ──
        if table_exists_in_supabase(supabase, table):
            choice = get_user_choice(table)
            if choice == "s":
                logger.log(f"Tabla '{table}' saltada por el usuario.", "SKIP")
                results["skipped"] += 1
                continue
            elif choice == "o":
                truncate_table(supabase, table, logger)
            # choice == "a" → simplemente sigue e inserta (append)
        else:
            # ── Crear tabla ──
            if not create_table_via_sql(supabase, table, schema, logger):
                results["errors"] += 1
                continue

        # ── Subir datos en lotes ──
        if row_count == 0:
            logger.log(f"Tabla '{table}' está vacía, nada que subir.", "OK")
            results["success"] += 1
            continue

        uploaded = 0
        num_batches = math.ceil(row_count / BATCH_SIZE)

        with tqdm(
            total=row_count,
            desc=f"  ⬆ {table}",
            unit=" filas",
            bar_format="{l_bar}{bar:30}{r_bar}",
            colour="green",
        ) as pbar:
            for batch in read_rows_in_batches(sqlite_conn, table, col_names, BATCH_SIZE):
                count = upload_batch(supabase, table, batch, logger)
                uploaded += count
                pbar.update(count)
                if count == 0:
                    pbar.colour = "red"

        if uploaded == row_count:
            logger.log(f"'{table}' completada: {uploaded:,}/{row_count:,} filas.", "OK")
            results["success"] += 1
        else:
            logger.log(f"'{table}' incompleta: {uploaded:,}/{row_count:,} filas.", "ERROR")
            results["errors"] += 1

        results["total_rows"] += uploaded

    # ── Resumen final ──
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    results["elapsed"] = f"{mins}m {secs}s"

    sqlite_conn.close()
    logger.summary(results)
    logger.log(f"Log completo guardado en: {LOG_FILE}", "INFO")


if __name__ == "__main__":
    migrate()
