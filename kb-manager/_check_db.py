import asyncio
from kb_manager.config import load_config
from kb_manager.models.database import Database, Document, Chunk
from sqlalchemy import func, select

async def check():
    cfg = load_config()
    print("DB URL:", cfg.db.async_url)
    db = Database(cfg.db)
    
    # Check the actual engine URL
    engine = db.async_engine
    print("Engine URL:", engine.url)
    
    async with db.session() as session:
        doc_count = (await session.execute(select(func.count(Document.id)))).scalar() or 0
        chunk_count = (await session.execute(select(func.count(Chunk.id)))).scalar() or 0
        print("doc_count:", doc_count)
        print("chunk_count:", chunk_count)
        
        # Also try fetching first doc
        result = await session.execute(select(Document).limit(1))
        first = result.scalar_one_or_none()
        if first:
            print("First doc:", first.id, first.title)
        else:
            print("NO DOCUMENTS FOUND!")
    
    await db.close()

asyncio.run(check())
