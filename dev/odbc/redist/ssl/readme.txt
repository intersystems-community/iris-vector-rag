For Unix/VMS platform SSL configuration.

irisodbc.ini.template is an example of how a odbc.ini file would be configured. IRIS examples use the file <irisinstalldir>/mgr/irisodbc.ini.  

odbcssl.ini.template is an example of a ssl configuration file.

The "#" entries must be removed from both template files and are only there for documentation purposes.


To specify an SSL connection: 
The environment variable ISC_SSLconfigurations must be defined for the process initiating the connection. The value of this variable must be a file that contains one or more named configurations that can be used to initiate an SSL connection. 
