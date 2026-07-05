import bcrypt
import jwt
import datetime
import uuid
import logging
from flask import current_app
from .db import db
from bson import ObjectId
from .config import Config

SECRET_KEY = Config.SECRET_KEY  # use the same secret from config

# ---------- Email integration with SendGrid ----------
try:
    import sendgrid
    from sendgrid.helpers.mail import Mail
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False
    logging.warning("SendGrid not installed. Email sending disabled.")

def send_reset_email(email, token):
    """Send a password reset email using SendGrid."""
    if not SENDGRID_AVAILABLE:
        logging.error("SendGrid not available")
        return False
    sg = sendgrid.SendGridAPIClient(api_key=Config.SENDGRID_API_KEY)
    reset_link = f"https://your-frontend-domain.com/reset-password?token={token}"
    message = Mail(
        from_email=Config.FROM_EMAIL,
        to_emails=email,
        subject="Password Reset Request",
        html_content=f"""
        <p>You requested a password reset.</p>
        <p>Click the link below to reset your password:</p>
        <a href='{reset_link}'>{reset_link}</a>
        <p>This link will expire in 24 hours.</p>
        """
    )
    try:
        response = sg.send(message)
        return response.status_code == 202
    except Exception as e:
        logging.error(f"SendGrid error: {e}")
        return False

# ---------- Password hashing ----------
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(hashed, password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# ---------- JWT ----------
def generate_token(user_id):
    payload = {
        'user_id': str(user_id),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def decode_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload.get('user_id')
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# ---------- User management ----------
def register_user(username, email_or_phone, password):
    if db.users.find_one({'$or': [{'username': username}, {'email_or_phone': email_or_phone}]}):
        return None, "Username or email/phone already exists"
    hashed = hash_password(password)
    user_id = db.users.insert_one({
        'username': username,
        'email_or_phone': email_or_phone,
        'password': hashed,
        'is_premium': False,
        'subscription_expiry': None,
        'subscription_plan': None,
        'created_at': datetime.datetime.utcnow(),
        'is_admin': False
    }).inserted_id
    return user_id, "User registered successfully"

def login_user(username_or_phone, password):
    user = db.users.find_one({'$or': [{'username': username_or_phone}, {'email_or_phone': username_or_phone}]})
    if not user:
        return None, "User not found"
    if not check_password(user['password'], password):
        return None, "Invalid password"
    token = generate_token(user['_id'])
    return token, "Login successful"

def get_user_by_id(user_id):
    return db.users.find_one({'_id': ObjectId(user_id)})

def is_premium(user_id):
    user = get_user_by_id(user_id)
    if not user:
        return False
    if user.get('is_premium') and user.get('subscription_expiry') and user['subscription_expiry'] > datetime.datetime.utcnow():
        return True
    # If expired, set is_premium to False
    if user.get('is_premium') and user.get('subscription_expiry') and user['subscription_expiry'] <= datetime.datetime.utcnow():
        db.users.update_one({'_id': ObjectId(user_id)}, {'$set': {'is_premium': False}})
    return False

def is_admin(user_id):
    user = get_user_by_id(user_id)
    return user and user.get('is_admin', False)

# ---------- Subscription ----------
def create_subscription_request(user_id, plan):
    plans = {'2weeks': 1000, '1month': 2500, '1year': 30000, 'forever': 100000}
    if plan not in plans:
        return None, "Invalid plan"
    user = get_user_by_id(user_id)
    if user and user.get('is_premium'):
        current_plan = user.get('subscription_plan')
        if current_plan and plans.get(current_plan, 0) > plans[plan]:
            return None, "Cannot downgrade to a lower plan while subscription is active"
    existing = db.subscription_requests.find_one({'user_id': ObjectId(user_id), 'plan': plan, 'status': 'pending'})
    if existing:
        return None, "You already have a pending request for this plan"
    db.subscription_requests.insert_one({
        'user_id': ObjectId(user_id),
        'plan': plan,
        'amount': plans[plan],
        'status': 'pending',
        'created_at': datetime.datetime.utcnow()
    })
    return "Subscription request submitted", None

def approve_subscription(user_id):
    user = get_user_by_id(user_id)
    if not user:
        return "User not found"
    request = db.subscription_requests.find_one({'user_id': ObjectId(user_id), 'status': 'pending'})
    if not request:
        return "No pending request"
    plan = request['plan']
    expiry = None
    if plan == '2weeks':
        expiry = datetime.datetime.utcnow() + datetime.timedelta(days=14)
    elif plan == '1month':
        expiry = datetime.datetime.utcnow() + datetime.timedelta(days=30)
    elif plan == '1year':
        expiry = datetime.datetime.utcnow() + datetime.timedelta(days=365)
    elif plan == 'forever':
        expiry = datetime.datetime.utcnow() + datetime.timedelta(days=365*100)
    db.users.update_one({'_id': ObjectId(user_id)}, {
        '$set': {
            'is_premium': True,
            'subscription_expiry': expiry,
            'subscription_plan': plan
        }
    })
    db.subscription_requests.update_one({'_id': request['_id']}, {'$set': {'status': 'approved'}})
    return "Subscription approved"

def decline_subscription(user_id):
    request = db.subscription_requests.find_one({'user_id': ObjectId(user_id), 'status': 'pending'})
    if not request:
        return "No pending request"
    db.subscription_requests.update_one({'_id': request['_id']}, {'$set': {'status': 'declined'}})
    return "Subscription declined"

def revoke_subscription(user_id):
    db.users.update_one({'_id': ObjectId(user_id)}, {'$set': {'is_premium': False, 'subscription_expiry': None, 'subscription_plan': None}})
    return "Subscription revoked"

def get_all_subscription_requests():
    return list(db.subscription_requests.find().sort('created_at', -1))

def get_users_with_expired_subscriptions():
    now = datetime.datetime.utcnow()
    return list(db.users.find({
        'is_premium': True,
        'subscription_expiry': {'$lt': now}
    }))

def expire_all_expired():
    users = get_users_with_expired_subscriptions()
    for user in users:
        db.users.update_one({'_id': user['_id']}, {'$set': {'is_premium': False}})
    return len(users)

def get_analytics():
    total_users = db.users.count_documents({})
    premium_users = db.users.count_documents({'is_premium': True})
    total_requests = db.subscription_requests.count_documents({})
    pending = db.subscription_requests.count_documents({'status': 'pending'})
    approved = db.subscription_requests.count_documents({'status': 'approved'})
    declined = db.subscription_requests.count_documents({'status': 'declined'})
    return {
        'total_users': total_users,
        'premium_users': premium_users,
        'total_requests': total_requests,
        'pending': pending,
        'approved': approved,
        'declined': declined
    }

# ---------- Password reset ----------
def request_password_reset(email_or_phone):
    user = db.users.find_one({'email_or_phone': email_or_phone})
    if not user:
        return None, "User not found"
    token = str(uuid.uuid4())
    db.password_reset_tokens.insert_one({
        'user_id': user['_id'],
        'token': token,
        'created_at': datetime.datetime.utcnow(),
        'expires_at': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    })
    return token, "Password reset token generated"

def reset_password(token, new_password):
    record = db.password_reset_tokens.find_one({'token': token})
    if not record or record['expires_at'] < datetime.datetime.utcnow():
        return "Invalid or expired token"
    hashed = hash_password(new_password)
    db.users.update_one({'_id': record['user_id']}, {'$set': {'password': hashed}})
    db.password_reset_tokens.delete_one({'_id': record['_id']})
    return "Password updated"

# ---------- Admin credentials (hardcoded) ----------
ADMIN_USERNAME = "Obasi excellent"
ADMIN_PASSWORD_HASH = hash_password("Excel1234@$")

def is_admin_credentials(username, password):
    return username == ADMIN_USERNAME and check_password(ADMIN_PASSWORD_HASH, password)