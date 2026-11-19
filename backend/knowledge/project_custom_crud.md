# Custom CRUD Application

A Django project built to go deliberately beyond built-in generic CRUD, into custom patterns, using Django and MySQL.

## Environment
Python 3.11.9, Django 5.2.17, and MySQL 8.0.46.

## Custom user system
Built with AbstractBaseUser and BaseUserManager.
- Fields: first name, last name, username (unique), email (unique), phone number, country name, address, and city.
- Login works with either the username or the email address, matched case-insensitively.

## Product CRUD
Fields: product name, price, category, brand, and product image, with image upload support. The list view shows columns for image, product name, price, category, brand, and actions.

## Project organization
Organized into templates, static, and media directories, with apps such as users and profile.
