import asyncio
from database import get_db, DocumentChunk, get_neo4j
from sqlalchemy import select

async def main():
    neo4j_driver = get_neo4j()
    
    with neo4j_driver.session() as session:
        records = session.run("MATCH (q:Question) WHERE q.image_url IS NOT NULL AND q.image_url <> '' RETURN q.course_code AS course, q.question_number AS q_num, q.image_url AS img").data()
        
    print(f"Found {len(records)} questions with images in Neo4j.")
    
    async for db in get_db():
        updated = 0
        for r in records:
            course = r['course']
            q_num = r['q_num']
            img = r['img']
            
            search_str = f"[PYQ - {course} - {q_num}]"
            stmt = select(DocumentChunk).where(DocumentChunk.content.contains(search_str))
            result = await db.execute(stmt)
            chunks = result.scalars().all()
            
            for chunk in chunks:
                if "![Diagram" not in chunk.content:
                    chunk.content += f"\n\n![Diagram for Q{q_num}]({img})"
                    updated += 1
                    
        await db.commit()
        print(f"Updated {updated} text chunks with images in PostgreSQL.")
        break

if __name__ == "__main__":
    asyncio.run(main())
