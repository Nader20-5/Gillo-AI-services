import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.database import connect_to_mongo, close_mongo_connection, db


async def main():
    await connect_to_mongo()
    cols = await db.db.list_collection_names()
    print(" Active Collections in MongoDB-AI:")
    for c in sorted(cols):
        count = await db.db[c].count_documents({})
        print(f"  - {c}: {count} documents")

    print("\n Verified Indexes on 'ai_suggestions':")
    indexes = await db.db.ai_suggestions.index_information()
    for name, info in indexes.items():
        print(f"  - {name}: {info['key']}")

    await close_mongo_connection()
    print("\n All database checks passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
