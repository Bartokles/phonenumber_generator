from phonenumbers import parse, is_valid_number
from phonenumbers.geocoder import description_for_number
from phonenumbers.carrier import name_for_number
from phonenumbers.phonenumberutil import PhoneNumberType, number_type

from concurrent.futures import ThreadPoolExecutor
import multiprocessing

from random import randint
import pandas as pd
import numpy as np

from pymysql import connect
from configparser import ConfigParser
from dotenv import load_dotenv
from os import getenv

from tqdm import tqdm
import logging


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_config(config, required_keys):
    missing = [key for key in required_keys if key not in config['default']]
    if missing:
        logger.error(f"Missing required config keys: {missing}")
        raise ValueError(f"Missing config keys: {', '.join(missing)}")


def get_table_name(config: ConfigParser, region_code: str) -> str:
    return config['regions'].get(region_code, None)


def build_dataframe(config: ConfigParser, numbers: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(numbers.size):
        try:
            phone_number = parse(f"{config['default']['regioncode']}{numbers[i]}", None)
            region_description = description_for_number(phone_number, config['default']['format'])
            carrier_name = name_for_number(phone_number, config['default']['format']).strip() or "Unknown"
            num_type = PhoneNumberType.to_string(number_type(phone_number))
        except Exception as e:
            logger.warning(f"Failed to parse number {numbers[i]}: {e}")
            continue

        rows.append({
            'NUMBER': numbers[i],
            'REGION': region_description,
            'CARRIER': carrier_name,
            'TYPE': num_type
        })

    return pd.DataFrame(rows)


def insert_data_to_mysql(data: pd.DataFrame, table_name: str, batch_size: int = 10000):
    DB_CONFIG = {
        'host': getenv('HOST'),
        'user': getenv('USER'),
        'password': getenv('PASSWORD'),
        'database': getenv('DATABASE'),
    }

    if not all(DB_CONFIG.values()):
        missing = [k for k, v in DB_CONFIG.items() if not v]
        logger.error(f"Missing database environment variables: {missing}")
        raise EnvironmentError(f"Missing DB config: {', '.join(missing)}")

    try:
        connection = connect(**DB_CONFIG)
        cursor = connection.cursor()

        columns = ", ".join(data.columns)
        values = ", ".join(["%s"] * len(data.columns))
        sql = f"INSERT IGNORE INTO {table_name} ({columns}) VALUES ({values})"

        total_inserted = 0
        for start in tqdm(range(0, len(data), batch_size), desc="Inserting to DB", unit="chunk"):
            batch = data.iloc[start:start + batch_size]
            rows = list(batch.itertuples(index=False, name=None))
            cursor.executemany(sql, rows)
            connection.commit()
            total_inserted += cursor.rowcount

        logger.info(f"Total inserted rows: {total_inserted}")

    except Exception as e:
        connection.rollback()
        logger.error(f"Database error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def remove_duplicates(numbers: np.ndarray) -> np.ndarray:
    unique_numbers = np.unique(numbers)
    logger.info(f"{len(numbers) - len(unique_numbers)} duplicates removed")
    return unique_numbers


def generate(length: int, regioncode: str) -> str:
    while True:
        number = "".join(str(randint(0, 9)) for _ in range(length))
        phone_number = parse(f"{regioncode}{number}", None)
        if is_valid_number(phone_number):
            return number


def main():
    load_dotenv()

    config = ConfigParser()
    config.read('config.ini')
    validate_config(config, ['regioncode', 'format', 'digits'])

    try:
        amount_of_numbers = int(input("How many numbers to generate: "))
        if amount_of_numbers <= 0:
            logger.warning("The number must be greater than 0")
            return

        numbers = np.empty(amount_of_numbers, dtype=object)

        logger.info("Starting number generation")

        max_threads = min(32, multiprocessing.cpu_count() * 2)
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            numbers = list(tqdm(
                executor.map(generate, [int(config['default']['digits'])] * amount_of_numbers,
                                    [config['default']['regioncode']] * amount_of_numbers),
                total=amount_of_numbers,
                desc="Generating numbers",
                unit="numbers"
            ))
            numbers = np.array(numbers, dtype=object)

        numbers = remove_duplicates(numbers)

        logger.info("Building in-memory dataset")
        phone_data = build_dataframe(config, numbers)

        if phone_data.empty:
            logger.warning("No valid phone numbers to insert.")
            return
        else:
            logger.info("Sending data to database")
            insert_data_to_mysql(phone_data, get_table_name(config, config['default']['regioncode']))
        
        #logger.info(f"Generated: {amount_of_numbers}, Unique: {len(numbers)}, Inserted: {len(phone_data)}") #FIX

    except KeyboardInterrupt:
        logger.warning("Program interrupted by user")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == '__main__':
    main()
