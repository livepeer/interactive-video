import argparse

from common import create_database
from models import Base


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-H", "--Host", help = "Database Host Domain/IP")
parser.add_argument("-P", "--Port", help = "Database Port")
parser.add_argument("-d", "--Dbname", help = "Database Name")
parser.add_argument("-u", "--Username", help = "Database Username")
parser.add_argument("-p", "--Password", help = "Database Password")
 
# Read arguments from command line
args = parser.parse_args()


if __name__ == '__main__':
    if args.Host is None:
        print('-H/--Host: expected one argument')
        exit()
        
    if args.Port is None:
        print('-P/--Port: expected one argument')
        exit()

    if args.Dbname is None:
        print('-d/--Dbname: expected one argument')
        exit()

    if args.Username is None:
        print('-u/--Username: expected one argument')
        exit()

    if args.Password is None:
        print('-p/--Password: expected one argument')
        exit()

    db_config = {
        'host': args.Host,
        'port': args.Port,
        'db_name': args.Dbname,
        'username': args.Username,
        'password': args.Password
    }

    res = create_database(model_base=Base, db_config=db_config)

    if res:
        print('Database has been initialized successfully')
    else:
        print('Database initialization failed')
