from pydantic import BaseModel
from datetime import datetime
from typing import List


class UserGet(BaseModel):
    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int
    os: str
    source: str

    class Config:
        orm_mode = True


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class FeedGet(BaseModel):
    user: UserGet
    post: PostGet
    user_id: int
    post_id: int
    action: str
    time: datetime

    class Config:
        orm_mode = True


