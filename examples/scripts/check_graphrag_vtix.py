import iris
import sys

try:
    # Connect to IRIS database
    connection = iris.connect(
        hostname='localhost',
        port=21972,
        namespace='GRAPHRAG',
        username='SuperUser',
        password='SYS'
    )
    
    cursor = connection.cursor()
    
    # Get entity statistics
    cursor.execute("""
        SELECT entity_type, COUNT(*) as entity_count
        FROM GraphRAG.Entity
        GROUP BY entity_type
        ORDER BY entity_count DESC
        LIMIT 10
    """)
    
    print("\n" + "="*60)
    print("📊 GRAPHRAG VALIDATION STATUS (vtix)")
    print("="*60)
    
    entity_types = cursor.fetchall()
    total_entities = sum(row[1] for row in entity_types)
    
    print(f"\n✅ Total Entities: {total_entities:,}")
    print("\nTop Entity Types:")
    for entity_type, count in entity_types:
        print(f"  • {entity_type}: {count:,}")
    
    # Get unique ticket count
    cursor.execute("""
        SELECT COUNT(DISTINCT source_document_id) as unique_tickets
        FROM GraphRAG.Entity
        WHERE source_document_id IS NOT NULL
        AND source_document_id != ''
    """)
    unique_tickets = cursor.fetchone()[0]
    
    print(f"\n🎟️  Unique Tickets Indexed: {unique_tickets:,} of 8,051 ({unique_tickets/8051*100:.1f}%)")
    
    # Get relationship count
    cursor.execute("SELECT COUNT(*) FROM GraphRAG.Relationship")
    rel_count = cursor.fetchone()[0]
    print(f"🔗 Total Relationships: {rel_count:,}")
    
    # Calculate average entities per ticket
    if unique_tickets > 0:
        avg_entities = total_entities / unique_tickets
        print(f"📈 Avg Entities/Ticket: {avg_entities:.1f}")
    
    cursor.close()
    connection.close()
    
    print("\n" + "="*60)
    print("✅ Status: INDEXING IN PROGRESS")
    print("="*60 + "\n")

except Exception as e:
    print(f"❌ Error connecting to IRIS: {e}")
    sys.exit(1)
