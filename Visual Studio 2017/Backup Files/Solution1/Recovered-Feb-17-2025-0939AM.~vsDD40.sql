-- Create the database
CREATE DATABASE ScrapMetalDB;
USE ScrapMetalDB;

-- Table for company details
CREATE TABLE Company (
    company_id INT identity(1,1) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    registration_number VARCHAR(50) UNIQUE NOT NULL,
    address TEXT,
    phone VARCHAR(20),
    email VARCHAR(100),
    website VARCHAR(100),
    operating_hours VARCHAR(100)
);

-- Table for scrap metal types
CREATE TABLE ScrapMetal (
    metal_id INT identity(1,1) PRIMARY KEY,
    metal_name VARCHAR(100) NOT NULL,
    category VARCHAR(20) CHECK (category IN ('Ferrous', 'Non-Ferrous', 'E-Waste', 'Specialty')) NOT NULL,
    description TEXT
);

-- Table for suppliers
CREATE TABLE Suppliers (
    supplier_id INT identity(1,1) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    contact_person VARCHAR(255),
    phone VARCHAR(20),
    email VARCHAR(100),
    address TEXT,
    company_name VARCHAR(255),
    metal_supplied INT,
    FOREIGN KEY (metal_supplied) REFERENCES ScrapMetal(metal_id)
);

-- Table for inventory (available scrap metal stock)
CREATE TABLE Inventory (
    inventory_id INT identity(1,1) PRIMARY KEY,
    metal_id INT,
    quantity_kg DECIMAL(10,2) NOT NULL,
    location VARCHAR(255),
    last_updated DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (metal_id) REFERENCES ScrapMetal(metal_id)
);

-- Table for pricing
CREATE TABLE Pricing (
    price_id INT IDENTITY(1,1) PRIMARY KEY,
    metal_id INT,
    buy_price DECIMAL(10,2) NOT NULL,
    sell_price DECIMAL(10,2) NOT NULL,
    last_updated DATETIME,
    FOREIGN KEY (metal_id) REFERENCES ScrapMetal(metal_id)
);

-- Table for customers/buyers
CREATE TABLE Customers (
    customer_id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    contact_person VARCHAR(255),
    phone VARCHAR(20),
    email VARCHAR(100),
    address TEXT,
    company_name VARCHAR(255),
    metal_purchased INT,
    FOREIGN KEY (metal_purchased) REFERENCES ScrapMetal(metal_id)
);

-- Table for transactions (buying & selling records)
CREATE TABLE Transactions (
    transaction_id INT IDENTITY(1,1) PRIMARY KEY,
    metal_id INT,
    supplier_id INT,
    customer_id INT,
    transaction_type VARCHAR(10) CHECK (transaction_type IN ('Purchase', 'Sale')) NOT NULL,
    quantity_kg DECIMAL(10,2) NOT NULL,
    price_per_kg DECIMAL(10,2) NOT NULL,
    total_price DECIMAL(10,2) NOT NULL,
    transaction_date DATETIME DEFAULT GETDATE(),
    FOREIGN KEY (metal_id) REFERENCES ScrapMetal(metal_id),
    FOREIGN KEY (supplier_id) REFERENCES Suppliers(supplier_id),
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);
