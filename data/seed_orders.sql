-- Campus Shop mock database seed

CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    items TEXT NOT NULL,
    tracking_number TEXT,
    order_date TEXT NOT NULL,
    total_amount REAL NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE IF NOT EXISTS escalations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reason TEXT NOT NULL,
    order_id INTEGER,
    customer_email TEXT,
    transcript TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR REPLACE INTO customers (id, name, email) VALUES
    (1, 'Aisha Khan', 'aisha.khan@student.buid.ac.ae'),
    (2, 'Omar Hassan', 'omar.hassan@student.buid.ac.ae'),
    (3, 'Sara Al Maktoum', 'sara.almaktoum@student.buid.ac.ae'),
    (4, 'James Wilson', 'james.wilson@student.buid.ac.ae'),
    (5, 'Fatima Noor', 'fatima.noor@student.buid.ac.ae');

INSERT OR REPLACE INTO orders (id, customer_id, status, items, tracking_number, order_date, total_amount) VALUES
    (1042, 1, 'delivered', 'Laptop Pro 14" (1x)', 'CS-TRK-8821', '2026-05-28', 3499.00),
    (1043, 2, 'shipped', 'Wireless Mouse (2x), USB-C Hub (1x)', 'CS-TRK-8822', '2026-06-01', 189.50),
    (1044, 3, 'processing', 'Campus Backpack (1x)', NULL, '2026-06-04', 129.00),
    (1045, 4, 'delivered', 'Noise-Cancelling Headphones (1x)', 'CS-TRK-8810', '2026-05-20', 599.00),
    (1046, 5, 'cancelled', 'Tablet 10" (1x)', NULL, '2026-05-15', 899.00),
    (1047, 1, 'shipped', 'Laptop Sleeve (1x)', 'CS-TRK-8830', '2026-06-03', 45.00),
    (1048, 2, 'delivered', 'External SSD 1TB (1x)', 'CS-TRK-8805', '2026-05-10', 279.00);
