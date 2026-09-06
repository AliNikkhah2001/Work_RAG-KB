import asyncio, pathlib
from sqlalchemy import text
from kb_manager.config import load_config
from kb_manager.models.database import Database

async def main():
    cfg = load_config()
    db = Database(cfg.db)
    async with db.session() as s:
        r = await s.execute(text("SELECT source_path FROM documents ORDER BY source_path"))
        rows = [row[0] for row in r.fetchall()]
        print(f"Documents in DB: {len(rows)}")
        for p in rows:
            print(repr(p))
        print("\n--- Checking one QA file ---")
        qpath = r"D:\Code\KB\kb-source\1405-05-31\(done)حقوقی\Company_CRM_Questions(done).xlsx"
        import pathlib as pl
        qnorm = str(pl.Path(qpath).resolve()).replace("\\","/")
        print("Looking for:", qnorm)
        for dbp in rows:
            if dbp.replace("\\","/").lower() == qnorm.lower():
                print("FOUND via normalized")
                break
        else:
            print("NOT FOUND")
            print("DB sample:", rows[0][:80] if rows else "none")

if __name__ == "__main__":
    asyncio.run(main())
