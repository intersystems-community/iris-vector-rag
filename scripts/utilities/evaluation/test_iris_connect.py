from iris_rag.config.manager import ConfigurationManager

print("Attempting to import iris...")
try:
    import iris
    print("Successfully imported 'iris' module.")
    print(f"Location of imported 'iris' module: {iris.__file__ if hasattr(iris, '__file__') else 'Unknown (built-in or no __file__ attribute)'}")
    
    if hasattr(iris, 'connect'):
        print("'iris' module HAS 'connect' attribute.")
        print("Attempting to get DBAPI connection details from ConfigurationManager...")
        
        # Initialize ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Fetch connection parameters using ConfigurationManager
        host = config_manager.get("database:iris:host", "localhost")
        port = config_manager.get("database:iris:port", 1972)
        namespace = config_manager.get("database:iris:namespace", "USER")
        user = config_manager.get("database:iris:user", "_SYSTEM")
        password = config_manager.get("database:iris:password", "SYS")
        
        # Ensure port is an integer
        if isinstance(port, str):
            port = int(port)
        
        print(f"Connection params: HOST={host}, PORT={port}, NAMESPACE={namespace}, USER={user}")
        
        try:
            print("Attempting iris.connect(...) with ssl=False")
            conn = iris.connect(host, port, namespace, user, password, ssl=False)
            print("Successfully connected using iris.connect() with ssl=False!")
            conn.close()
            print("Connection closed.")
        except Exception as e_ssl_false:
            print(f"iris.connect() with ssl=False FAILED: {e_ssl_false}")
            try:
                print("Attempting iris.connect(...) without ssl parameter")
                conn = iris.connect(host, port, namespace, user, password)
                print("Successfully connected using iris.connect() without ssl parameter!")
                conn.close()
                print("Connection closed.")
            except Exception as e_no_ssl:
                print(f"iris.connect() without ssl parameter FAILED: {e_no_ssl}")
                
    else:
        print("'iris' module DOES NOT HAVE 'connect' attribute.")
        print("Attempting to import iris.dbapi as fallback1...")
        try:
            import iris.dbapi as irisdbapi_alt
            print(f"Location of imported 'iris': {irisdbapi_alt.__file__ if hasattr(irisdbapi_alt, '__file__') else 'Unknown'}")
            if hasattr(irisdbapi_alt, 'connect'):
                print("Successfully imported 'iris' and it HAS 'connect'.")
            else:
                print("Imported 'iris' but it DOES NOT HAVE 'connect'.")
        except ImportError as e_alt:
            print(f"Failed to import 'iris': {e_alt}")
            print("Attempting to import irisnative.dbapi as fallback2...")
            try:
                import iris as irisdbapi_native
                print(f"Location of imported 'irisnative.dbapi': {irisdbapi_native.__file__ if hasattr(irisdbapi_native, '__file__') else 'Unknown'}")
                if hasattr(irisdbapi_native, 'connect'):
                    print("Successfully imported 'irisnative.dbapi' and it HAS 'connect'.")
                else:
                    print("Imported 'irisnative.dbapi' but it DOES NOT HAVE 'connect'.")
            except ImportError as e_native:
                print(f"Failed to import 'irisnative.dbapi': {e_native}")

except ImportError as e:
    print(f"Failed to import 'iris' module: {e}")
except Exception as e_general:
    print(f"An unexpected error occurred: {e_general}")

print("Test script finished.")