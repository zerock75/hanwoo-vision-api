import aiomysql

pool: aiomysql.Pool | None = None

from hanwoo.core.config import DB_233_HOST, DB_233_PORT, DB_233_USER, DB_233_PASSWORD, DB_233_NAME

async def db_233():
    global pool
    pool = await aiomysql.create_pool(
        host=DB_233_HOST,
        port=DB_233_PORT,
        user=DB_233_USER,
        password=DB_233_PASSWORD,
        db=DB_233_NAME,
        autocommit=True,
    )

