CREATE DATABASE st_project_1;

USE st_project_1;

CREATE TABLE walmart_dataset (
    invoice_id      INT PRIMARY KEY,
    Branch          VARCHAR(10),
    City            VARCHAR(50),
    category        VARCHAR(50),
    unit_price      varchar(20),
    quantity        INT,
    date            VARCHAR(20),
    time            VARCHAR(20),
    payment_method  VARCHAR(20),
    rating          DECIMAL(3,1),
    profit_margin   DECIMAL(5,2)
);

select * from walmart_cleaned_dataset;

