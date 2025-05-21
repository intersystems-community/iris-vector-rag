import os
import sys
from sqlalchemy import create_engine, text, Column, Integer, MetaData, Table, select
from sqlalchemy.orm import Session
try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base # type: ignore
    
# Attempt to import IRISVectorType from sqlalchemy_iris
try:
    from sqlalchemy_iris import IRISVector # This is the type for defining columns
except ImportError:
    print("ERROR: sqlalchemy-iris library not found. Please install it.")
    print("Try: pip install sqlalchemy-iris")
    sys.exit(1)

# --- Connection Parameters ---
# Default connection string for intersystems-iris DBAPI driver (iris+emb)
# Assumes IRIS instance is available locally with default settings, connecting to USER namespace.
# You might need to adjust if your IRIS instance or credentials differ.
# Example: "iris://username:password@hostname:port/namespace"
# Example for docker default: "iris://_SYSTEM:SYS@localhost:1972/USER"
DEFAULT_IRIS_CONNECTION_STRING = "iris://SuperUser:SYS@localhost:1972/USER" # Changed to explicit network string
IRIS_CONNECTION_STRING = os.getenv("IRIS_CONNECTION_STRING_SQLALCHEMY", DEFAULT_IRIS_CONNECTION_STRING)

TABLE_NAME = "RAG_SQLAlchemyVectorTest"
VECTOR_DIM = 3
Base = declarative_base()

# Define the table using SQLAlchemy ORM (or use Core Table definition)
class VectorTestTable(Base): # type: ignore
    __tablename__ = TABLE_NAME
    __table_args__ = {'extend_existing': True, "schema": "RAG"} # Using RAG schema

    id = Column(Integer, primary_key=True)
    embedding = Column(IRISVector(VECTOR_DIM)) # Reverted: IRISVector does not take element_type

def main():
    print(f"Attempting to connect to IRIS using SQLAlchemy with connection string: {IRIS_CONNECTION_STRING}")
    engine = None
    try:
        # Enable SQL echoing to see generated queries
        engine = create_engine(IRIS_CONNECTION_STRING, echo=True) 
        with engine.connect() as connection:
            print("Successfully connected to IRIS via SQLAlchemy.")
            
            # Ensure RAG schema exists (optional, depends on DB setup)
            try:
                connection.execute(text("CREATE SCHEMA RAG"))
                connection.commit()
                print("Schema RAG ensured or created.")
            except Exception as e_schema:
                print(f"Note: Could not create RAG schema (it might already exist): {e_schema}")
                # If using implicit transactions, a rollback might be needed if schema creation failed within a transaction
                try:
                    connection.rollback()
                except: # noqa
                    pass


            metadata = Base.metadata
            
            # Drop table if it exists (for a clean test run)
            print(f"Dropping table {VectorTestTable.__table_args__['schema']}.{TABLE_NAME} if it exists...")
            VectorTestTable.__table__.drop(connection, checkfirst=True) # Using the schema
            connection.commit() # Commit drop if not auto-committed by DDL

            # Create the table
            print(f"Creating table {VectorTestTable.__table_args__['schema']}.{TABLE_NAME}...")
            metadata.create_all(connection)
            connection.commit() # Commit create if not auto-committed by DDL
            print("Table created successfully.")

            # Insert data
            print("\nAttempting to insert data...")
            test_id = 1
            test_vector_py_list = [0.1, 0.2, 0.3]
            
            with Session(connection) as session:
                new_entry = VectorTestTable(id=test_id, embedding=test_vector_py_list)
                session.add(new_entry)
                session.commit()
            print(f"Data inserted: id={test_id}, embedding={test_vector_py_list}")

            # Query data using cosine similarity
            print("\nAttempting to query data using vector_cosine similarity...")
            query_vector_py_list = [0.1, 0.2, 0.3] # Perfect match

            with Session(connection) as session:
                # Using explicit func.vector_cosine and text(TO_VECTOR(...))
                # This was the version from 7:01 AM that generated correct SQL but still failed at driver level
                from sqlalchemy import func, text

                inner_vector_content = f"[{','.join(map(str, query_vector_py_list))}]"
                to_vector_argument_sql = f"TO_VECTOR('{inner_vector_content}', 'DOUBLE', {VECTOR_DIM})"
                
                stmt = (
                    select(
                        VectorTestTable.id,
                        VectorTestTable.embedding,
                        func.vector_cosine(
                            VectorTestTable.embedding,
                            text(to_vector_argument_sql) 
                        ).label("score")
                    )
                    .order_by(text("score DESC"))
                )
                
                print(f"Executing SQLAlchemy query for similarity with inlined TO_VECTOR string...")
                result = session.execute(stmt).fetchone()

            if result:
                print("\nQuery Results:")
                print(f"  ID: {result.id}")
                print(f"  Embedding (from DB, type {type(result.embedding)}): {result.embedding}")
                print(f"  Calculated Score: {result.score}")

                if result.embedding and isinstance(result.embedding, list) and \
                   all(isinstance(x, float) for x in result.embedding) and \
                   len(result.embedding) == VECTOR_DIM:
                    print("  Embedding type from DB is Python list of floats as expected.")
                else:
                    print("WARNING: Embedding type from DB is NOT a Python list of floats as expected.")

                if result.score is not None and abs(result.score - 1.0) < 1e-6:
                    print("SUCCESS: Cosine similarity with query vector returned expected score of ~1.0.")
                else:
                    print(f"WARNING: Score was {result.score}, expected ~1.0.")
            else:
                print("ERROR: No rows returned from select query.")

    except ImportError:
        # This was already checked at the top, but as a fallback.
        print("ERROR: sqlalchemy-iris library not found. Please install it.")
        return 1
    except Exception as e:
        print(f"\nAn error occurred:")
        print(e)
        # import traceback
        # traceback.print_exc() # For more detailed errors if needed
        return 1
    finally:
        if engine:
            # Clean up: Drop the test table
            try:
                with engine.connect() as connection:
                    print(f"\nSKIPPING table drop for manual inspection: {VectorTestTable.__table_args__['schema']}.{TABLE_NAME}...")
                    # VectorTestTable.__table__.drop(connection, checkfirst=True) # Commented out for manual inspection
                    # connection.commit() # No longer needed if drop is commented
                    print("Table drop SKIPPED for manual inspection.")
            except Exception as e_cleanup:
                print(f"Error during cleanup: {e_cleanup}")
            engine.dispose()
            print("SQLAlchemy engine disposed.")
            
    return 0

if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        print("\nSQLAlchemy vector operations test completed.")
    else:
        print("\nSQLAlchemy vector operations test FAILED.")
    sys.exit(exit_code)
