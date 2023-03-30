from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context



# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from models import *
import util.config
util.config.Config.init()

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = None

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Edited by rknop from 'alembic init' output for our database handling
    """

    with base.SmartSession() as session:
        url = config.get_main_option("sqlalchemy.url")
        context.configure(
            connection=session.connection(),
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
        )

        with context.begin_transaction():
            context.run_migrations()
            session.commit()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Edited by rknop from 'alembic init' output for our database handling
    """
    with base.SmartSession() as session:
        context.configure(
            connection=session.connection(),
            target_metadata=base.Base.metadata
        )

        with context.begin_transaction():
            context.run_migrations()
            session.commit()
            

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
