unixODBC Configuration
    A unixODBC installation should have installed the program odbcinst.
    	This may be located under /usr/local/bin (if unixODBC is installed after the OS).

	Two template files are included for unixODBC:
		odbc.ini_unixODBCtemplate - Sample DSN entry template
		odbcinst.ini_unixODBCtemplate - Intersystems driver template
	Edit the template files to suit your configuration


  	to use the template files first register the driver:
		# odbcinst -i -d -f odbcinst.ini_unixODBCtemplate

  	to add a Local DSN
		# odbcinst -i -s -h -f odbc.ini_unixODBCtemplate

  	to add a System DSN
		# odbcinst -i -s -l -f odbc.ini_unixODBCtemplate




