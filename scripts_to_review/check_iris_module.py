#!/usr/bin/env python
# check_iris_module.py
# A script to investigate the structure of the IRIS Python module

def main():
    print("Checking InterSystems IRIS module structure...")
    
    try:
        # First, try to import the module
        import intersystems_iris
        print(f"✅ Successfully imported intersystems_iris module")
        print(f"Module version: {getattr(intersystems_iris, '__version__', 'Unknown')}")
        print(f"Module path: {intersystems_iris.__file__}")
        print("\nModule contents:")
        
        # List the module's attributes and submodules
        import inspect
        attrs = dir(intersystems_iris)
        for attr in attrs:
            if not attr.startswith("__"):
                print(f"- {attr}")
        
        # Check the DBAPI submodule
        print("\nChecking DBAPI submodule...")
        if hasattr(intersystems_iris, 'dbapi'):
            import intersystems_iris.dbapi as dbapi
            print("✅ Successfully imported intersystems_iris.dbapi")
            print("\nDBAPI module contents:")
            
            dbapi_attrs = dir(dbapi)
            for attr in dbapi_attrs:
                if not attr.startswith("__"):
                    value = getattr(dbapi, attr)
                    if inspect.isclass(value):
                        print(f"- {attr} (Class)")
                    elif inspect.isfunction(value):
                        print(f"- {attr} (Function)")
                    else:
                        print(f"- {attr}")
            
            # Check if the Connection class exists
            if hasattr(dbapi, 'Connection'):
                conn_class = dbapi.Connection
                print("\nConnection class details:")
                print(f"Connection class: {conn_class}")
                
                # Try to get the signature
                try:
                    import inspect
                    signature = inspect.signature(conn_class.__init__)
                    print(f"Constructor signature: {signature}")
                except Exception as e:
                    print(f"Could not get signature: {e}")
        else:
            print("❌ intersystems_iris.dbapi submodule not found")
    
    except ImportError as e:
        print(f"❌ Could not import intersystems_iris module: {e}")
    except Exception as e:
        print(f"❌ Error inspecting module: {e}")

if __name__ == "__main__":
    main()
