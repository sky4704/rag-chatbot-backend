import argparse
import sys
import getpass
from utils.db import SessionLocal, User, init_db
from utils.auth import get_password_hash

def create_superuser():
    print("\n--- NexRead Admin Creation ---")
    username = input("Enter admin username: ")
    
    db = SessionLocal()
    init_db()
    
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        print(f"Error: User '{username}' already exists.")
        db.close()
        return

    password = getpass.getpass("Enter password: ")
    confirm_password = getpass.getpass("Confirm password: ")
    
    if password != confirm_password:
        print("Error: Passwords do not match.")
        db.close()
        return

    hashed_password = get_password_hash(password)
    new_user = User(username=username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    print(f"Success: Superuser '{username}' created.")
    db.close()

def change_password():
    print("\n--- Change Admin Password ---")
    username = input("Enter username: ")
    
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    
    if not user:
        print(f"Error: User '{username}' not found.")
        db.close()
        return

    new_password = getpass.getpass("Enter new password: ")
    confirm_password = getpass.getpass("Confirm new password: ")
    
    if new_password != confirm_password:
        print("Error: Passwords do not match.")
        db.close()
        return

    user.hashed_password = get_password_hash(new_password)
    db.commit()
    print(f"Success: Password updated for '{username}'.")
    db.close()

def list_users():
    db = SessionLocal()
    users = db.query(User).all()
    print("\nExisting Admins:")
    for u in users:
        print(f"- {u.username}")
    db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NexRead Management CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("createsuperuser", help="Create a new admin user interactively")
    subparsers.add_parser("changepassword", help="Change password for an existing admin")
    subparsers.add_parser("listusers", help="List all admin users")

    args = parser.parse_args()

    if args.command == "createsuperuser":
        create_superuser()
    elif args.command == "changepassword":
        change_password()
    elif args.command == "listusers":
        list_users()
    else:
        parser.print_help()
