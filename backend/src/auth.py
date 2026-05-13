import sqlite3
import os
import jwt
from datetime import datetime, timedelta
import bcrypt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("SECRET_KEY", "supersecret-graphrag-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week


security = HTTPBearer()

def get_db():
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect("db/users.db", check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT)")
    conn.commit()
    return conn

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid auth token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid auth token")
    return username
