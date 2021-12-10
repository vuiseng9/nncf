"""Wraps the 'flask' command to automatically set FLASK_APP and FLASK_DEBUG environment variable.
Typical usage:
manage.py run
or to initialize the database:
manage.py init_db
Make IDE debugging easier.
"""

from flask.cli import main
import os, sys


if __name__ == '__main__':
    os.environ['LC_ALL'] = 'C.UTF-8'
    os.environ['LANG'] = 'C.UTF-8'

    os.environ['FLASK_APP'] = 'qaas'
    os.environ['FLASK_DEBUG'] = 'true'

    print("main")
    # sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(os.path.dirname(__file__))))))
    main()