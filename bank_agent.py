from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


class DatabaseConn:
    """This is a fake database for example purposes.

     In reality, you'd be connecting to an external database
     (e.g. PostgreSQL) to get information about customers.
     """

#    @classmethod
#    async def customer_name(cls, *, id: int) -> str | None:
#         if id == 123:
#             return 'John'
    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return 'John'
