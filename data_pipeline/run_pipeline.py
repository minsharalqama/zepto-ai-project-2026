from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
BASELINE_GBP_TO_INR = 105.50
OUTPUT_DIR = Path(__file__).resolve().parent
DB_PATH = OUTPUT_DIR / "books.db"
RAW_CSV = OUTPUT_DIR / "scraped_books_raw.csv"
CLEAN_CSV = OUTPUT_DIR / "scraped_books_clean.csv"
QUERIES_OUTPUT = OUTPUT_DIR / "queries_output.md"

HEADERS = {"User-Agent": "Mozilla/5.0 (Zepto-AI-Capstone-Scraper)"}


def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def discover_categories() -> list[tuple[str, str]]:
    soup = get_soup(BASE_URL)
    items = []
    for link in soup.select(".side_categories ul li ul li a"):
        name = link.get_text(strip=True)
        href = link.get("href")
        if href:
            items.append((name, BASE_URL.rstrip("/") + "/" + href.lstrip("/")))
    return items


def scrape_category(category_name: str, category_url: str) -> list[dict]:
    rows = []
    next_url = category_url
    page_no = 1

    while next_url:
        soup = get_soup(next_url)
        products = soup.select("article.product_pod")
        for product in products:
            title_link = product.select_one("h3 a")
            price_el = product.select_one(".price_color")
            rating_el = product.select_one("p.star-rating")
            availability_listing = product.select_one(".availability")

            if not title_link:
                continue

            title = title_link.get("title") or title_link.get_text(strip=True)
            detail_href = title_link.get("href")
            detail_url = requests.compat.urljoin(next_url, detail_href)

            availability_text = None
            try:
                detail_soup = get_soup(detail_url)
                availability_el = detail_soup.select_one(".product_main .availability")
                availability_text = availability_el.get_text(" ", strip=True) if availability_el else None
            except requests.RequestException as exc:
                print(f"WARNING: detail page failed for {title!r}: {exc}")
                availability_text = (
                    availability_listing.get_text(" ", strip=True)
                    if availability_listing
                    else None
                )

            rows.append(
                {
                    "title": title,
                    "price": price_el.get_text(strip=True) if price_el else None,
                    "star_rating": " ".join(rating_el.get("class", [])[1:]) if rating_el else None,
                    "availability": availability_text,
                    "category": category_name,
                }
            )

        next_link = soup.select_one("li.next a")
        next_url = requests.compat.urljoin(next_url, next_link.get("href")) if next_link else None
        print(f"Scraped {category_name!r} page {page_no}: {len(products)} books")
        page_no += 1

    return rows


def scrape_at_least_60() -> pd.DataFrame:
    categories = discover_categories()
    if len(categories) < 3:
        raise RuntimeError("Could not discover at least three book categories.")

    selected = []
    rows: list[dict] = []
    for category_name, category_url in categories:
        selected.append(category_name)
        rows.extend(scrape_category(category_name, category_url))
        if len(rows) >= 60 and len(selected) >= 3:
            break

    raw_df = pd.DataFrame(rows).drop_duplicates(subset=["title", "category"])
    if len(raw_df) < 60:
        raise RuntimeError(f"Only {len(raw_df)} rows scraped; at least 60 are required.")
    if raw_df["category"].nunique() < 3:
        raise RuntimeError("At least three distinct categories are required.")
    return raw_df


def parse_price(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value).replace(",", ""))
    return float(match.group(1)) if match else float("nan")


def parse_rating(value: object) -> float:
    mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    return mapping.get(text, float("nan"))


def parse_stock(value: object) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if "in stock" in text:
        return True
    if "out of stock" in text:
        return False
    return None


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["price_gbp"] = df["price"].map(parse_price)
    df["rating"] = df["star_rating"].map(parse_rating)
    df["in_stock"] = df["availability"].map(parse_stock)

    for col in ["title", "category", "in_stock"]:
        before = len(df)
        df = df.dropna(subset=[col])
        print(f"Dropped {before - len(df)} rows because {col!r} could not be parsed.")

    for col in ["price_gbp", "rating"]:
        missing = int(df[col].isna().sum())
        if missing:
            median = float(df[col].median())
            df[col] = df[col].fillna(median)
            print(f"Imputed {missing} {col!r} parse failures with median={median:.3f}.")

    df["rating"] = df["rating"].round().clip(1, 5).astype(int)
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = (df["price_gbp"] * BASELINE_GBP_TO_INR).round(2)

    result = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]].copy()
    return result


def load_database(df: pd.DataFrame) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id) REFERENCES categories(category_id)
            );
            """
        )

        categories = sorted(df["category"].unique())
        for idx, category in enumerate(categories, start=1):
            conn.execute(
                "INSERT INTO categories(category_id, category_name) VALUES (?, ?)",
                (idx, category),
            )

        category_map = {name: idx for idx, name in enumerate(categories, start=1)}
        records = [
            (
                row.title,
                row.price_gbp,
                row.price_inr,
                int(row.rating),
                int(row.in_stock),
                category_map[row.category],
            )
            for row in df.itertuples(index=False)
        ]
        conn.executemany(
            """
            INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()


def run_queries() -> None:
    queries = {
        "Q1 - SELECT + WHERE": """
            SELECT title, rating, price_gbp, in_stock
            FROM books
            WHERE in_stock = 1 AND rating >= 4
            ORDER BY rating DESC, title
            LIMIT 10;
        """,
        "Q2 - ORDER BY + LIMIT": """
            SELECT title, price_gbp, price_inr
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10;
        """,
        "Q3 - DISTINCT": """
            SELECT DISTINCT c.category_name
            FROM categories c
            JOIN books b ON b.category_id = c.category_id
            ORDER BY c.category_name;
        """,
        "Q4 - BETWEEN": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE price_gbp BETWEEN 10 AND 20
            ORDER BY price_gbp;
        """,
        "Q5 - JOIN": """
            SELECT b.title, b.rating, b.price_gbp, c.category_name
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            ORDER BY b.rating DESC, b.price_gbp DESC, b.title
            LIMIT 10;
        """,
        "Q6 - JOIN + aggregation": """
            SELECT c.category_name,
                   COUNT(*) AS book_count,
                   ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
                   ROUND(AVG(b.price_inr), 2) AS avg_price_inr
            FROM categories c
            JOIN books b ON b.category_id = c.category_id
            GROUP BY c.category_id, c.category_name
            ORDER BY avg_price_gbp DESC;
        """,
    }

    md_parts = ["# SQL Query Outputs\n"]
    with sqlite3.connect(DB_PATH) as conn:
        sql_join_df = pd.read_sql_query(queries["Q5 - JOIN"], conn)
        books_df = pd.read_sql_query("SELECT * FROM books", conn)
        categories_df = pd.read_sql_query("SELECT * FROM categories", conn)

        pandas_join_df = (
            books_df.merge(categories_df, on="category_id", how="inner")
            [["title", "rating", "price_gbp", "category_name"]]
            .sort_values(["rating", "price_gbp", "title"], ascending=[False, False, True])
            .head(10)
            .reset_index(drop=True)
        )
        sql_join_comp = sql_join_df.reset_index(drop=True)
        equivalent = sql_join_comp.equals(pandas_join_df)

        for name, query in queries.items():
            result = pd.read_sql_query(query, conn)
            md_parts.append(f"## {name}\n")
            md_parts.append("```sql\n" + query.strip() + "\n```\n")
            md_parts.append(result.to_markdown(index=False))
            md_parts.append("\n")

        md_parts.append("## pd.read_sql JOIN vs pd.merge JOIN\n")
        md_parts.append("### SQL result (`pd.read_sql`)\n")
        md_parts.append(sql_join_comp.to_markdown(index=False))
        md_parts.append("\n### Pandas result (`pd.merge`)\n")
        md_parts.append(pandas_join_df.to_markdown(index=False))
        md_parts.append(f"\nEquivalent output: **{equivalent}**\n")

    QUERIES_OUTPUT.write_text("\n".join(md_parts), encoding="utf-8")
    print("SQL outputs saved to", QUERIES_OUTPUT)
    print("pd.read_sql JOIN equals pd.merge JOIN:", equivalent)


def main() -> None:
    print("1) Scraping books.toscrape.com")
    raw_df = scrape_at_least_60()
    raw_df.to_csv(RAW_CSV, index=False)
    print(f"Raw rows: {len(raw_df)}; categories: {raw_df['category'].nunique()}")
    print("Raw data saved to", RAW_CSV)

    print("\n2) Cleaning + conversion")
    clean_df = clean_data(raw_df)
    clean_df.to_csv(CLEAN_CSV, index=False)
    print(clean_df.dtypes)
    print(f"Clean rows: {len(clean_df)}; categories: {clean_df['category'].nunique()}")
    print(f"Fixed conversion baseline: 1 GBP = {BASELINE_GBP_TO_INR:.2f} INR")

    print("\n3) Loading normalized SQLite database")
    load_database(clean_df)
    print("Database saved to", DB_PATH)

    print("\n4) SQL queries + pandas equivalence check")
    run_queries()


if __name__ == "__main__":
    main()
