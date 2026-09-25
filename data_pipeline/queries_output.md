# SQL Query Outputs

## Q1 - SELECT + WHERE

```sql
SELECT title, rating, price_gbp, in_stock
            FROM books
            WHERE in_stock = 1 AND rating >= 4
            ORDER BY rating DESC, title
            LIMIT 10;
```

| title                                                                    |   rating |   price_gbp |   in_stock |
|:-------------------------------------------------------------------------|---------:|------------:|-----------:|
| 1,000 Places to See Before You Die                                       |        5 |       26.08 |          1 |
| A Flight of Arrows (The Pathfinders #2)                                  |        5 |       55.53 |          1 |
| A Spy's Devotion (The Regency Spies of London #1)                        |        5 |       16.97 |          1 |
| A Time of Torment (Charlie Parker #14)                                   |        5 |       48.35 |          1 |
| Between Shades of Gray                                                   |        5 |       20.79 |          1 |
| Mrs. Houdini                                                             |        5 |       30.25 |          1 |
| The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) |        5 |       52.3  |          1 |
| The Girl You Lost                                                        |        5 |       12.29 |          1 |
| The Passion of Dolssa                                                    |        5 |       28.32 |          1 |
| The Red Tent                                                             |        5 |       35.66 |          1 |


## Q2 - ORDER BY + LIMIT

```sql
SELECT title, price_gbp, price_inr
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10;
```

| title                                                                  |   price_gbp |   price_inr |
|:-----------------------------------------------------------------------|------------:|------------:|
| Boar Island (Anna Pigeon #19)                                          |       59.48 |     6275.14 |
| The No. 1 Ladies' Detective Agency (No. 1 Ladies' Detective Agency #1) |       57.7  |     6087.35 |
| A Year in Provence (Provence #1)                                       |       56.88 |     6000.84 |
| The Past Never Ends                                                    |       56.5  |     5960.75 |
| The Last Painting of Sara de Vos                                       |       55.55 |     5860.52 |
| A Flight of Arrows (The Pathfinders #2)                                |       55.53 |     5858.42 |
| Murder at the 42nd Street Library (Raymond Ambler #1)                  |       54.36 |     5734.98 |
| The Last Mile (Amos Decker #2)                                         |       54.21 |     5719.16 |
| 1st to Die (Women's Murder Club #1)                                    |       53.98 |     5694.89 |
| Tipping the Velvet                                                     |       53.74 |     5669.57 |


## Q3 - DISTINCT

```sql
SELECT DISTINCT c.category_name
            FROM categories c
            JOIN books b ON b.category_id = c.category_id
            ORDER BY c.category_name;
```

| category_name      |
|:-------------------|
| Historical Fiction |
| Mystery            |
| Travel             |


## Q4 - BETWEEN

```sql
SELECT title, price_gbp, rating
            FROM books
            WHERE price_gbp BETWEEN 10 AND 20
            ORDER BY price_gbp;
```

| title                                             |   price_gbp |   rating |
|:--------------------------------------------------|------------:|---------:|
| Tastes Like Fear (DI Marnie Rome #3)              |       10.69 |        1 |
| Hide Away (Eve Duncan #20)                        |       11.84 |        1 |
| The Girl You Lost                                 |       12.29 |        5 |
| Playing with Fire                                 |       13.71 |        3 |
| That Darkness (Gardiner and Renner #1)            |       13.92 |        1 |
| The Girl In The Ice (DCI Erika Foster #1)         |       15.85 |        3 |
| The Constant Princess (The Tudor Court #1)        |       16.62 |        3 |
| A Murder in Time                                  |       16.64 |        1 |
| A Study in Scarlet (Sherlock Holmes #1)           |       16.73 |        2 |
| A Spy's Devotion (The Regency Spies of London #1) |       16.97 |        5 |
| Lilac Girls                                       |       17.28 |        2 |
| The Cuckoo's Calling (Cormoran Strike #1)         |       19.21 |        1 |
| In a Dark, Dark Wood                              |       19.63 |        1 |


## Q5 - JOIN

```sql
SELECT b.title, b.rating, b.price_gbp, c.category_name
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            ORDER BY b.rating DESC, b.price_gbp DESC, b.title
            LIMIT 10;
```

| title                                                                    |   rating |   price_gbp | category_name      |
|:-------------------------------------------------------------------------|---------:|------------:|:-------------------|
| A Flight of Arrows (The Pathfinders #2)                                  |        5 |       55.53 | Historical Fiction |
| The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) |        5 |       52.3  | Mystery            |
| A Time of Torment (Charlie Parker #14)                                   |        5 |       48.35 | Mystery            |
| While You Were Mine                                                      |        5 |       41.32 | Historical Fiction |
| The Red Tent                                                             |        5 |       35.66 | Historical Fiction |
| Mrs. Houdini                                                             |        5 |       30.25 | Historical Fiction |
| The Passion of Dolssa                                                    |        5 |       28.32 | Historical Fiction |
| 1,000 Places to See Before You Die                                       |        5 |       26.08 | Travel             |
| What Happened on Beale Street (Secrets of the South Mysteries #2)        |        5 |       25.37 | Mystery            |
| The Silkworm (Cormoran Strike #2)                                        |        5 |       23.05 | Mystery            |


## Q6 - JOIN + aggregation

```sql
SELECT c.category_name,
                   COUNT(*) AS book_count,
                   ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
                   ROUND(AVG(b.price_inr), 2) AS avg_price_inr
            FROM categories c
            JOIN books b ON b.category_id = c.category_id
            GROUP BY c.category_id, c.category_name
            ORDER BY avg_price_gbp DESC;
```

| category_name      |   book_count |   avg_price_gbp |   avg_price_inr |
|:-------------------|-------------:|----------------:|----------------:|
| Travel             |           11 |           39.79 |         4198.32 |
| Historical Fiction |           26 |           33.64 |         3549.47 |
| Mystery            |           32 |           31.72 |         3346.36 |


## pd.read_sql JOIN vs pd.merge JOIN

### SQL result (`pd.read_sql`)

| title                                                                    |   rating |   price_gbp | category_name      |
|:-------------------------------------------------------------------------|---------:|------------:|:-------------------|
| A Flight of Arrows (The Pathfinders #2)                                  |        5 |       55.53 | Historical Fiction |
| The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) |        5 |       52.3  | Mystery            |
| A Time of Torment (Charlie Parker #14)                                   |        5 |       48.35 | Mystery            |
| While You Were Mine                                                      |        5 |       41.32 | Historical Fiction |
| The Red Tent                                                             |        5 |       35.66 | Historical Fiction |
| Mrs. Houdini                                                             |        5 |       30.25 | Historical Fiction |
| The Passion of Dolssa                                                    |        5 |       28.32 | Historical Fiction |
| 1,000 Places to See Before You Die                                       |        5 |       26.08 | Travel             |
| What Happened on Beale Street (Secrets of the South Mysteries #2)        |        5 |       25.37 | Mystery            |
| The Silkworm (Cormoran Strike #2)                                        |        5 |       23.05 | Mystery            |

### Pandas result (`pd.merge`)

| title                                                                    |   rating |   price_gbp | category_name      |
|:-------------------------------------------------------------------------|---------:|------------:|:-------------------|
| A Flight of Arrows (The Pathfinders #2)                                  |        5 |       55.53 | Historical Fiction |
| The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) |        5 |       52.3  | Mystery            |
| A Time of Torment (Charlie Parker #14)                                   |        5 |       48.35 | Mystery            |
| While You Were Mine                                                      |        5 |       41.32 | Historical Fiction |
| The Red Tent                                                             |        5 |       35.66 | Historical Fiction |
| Mrs. Houdini                                                             |        5 |       30.25 | Historical Fiction |
| The Passion of Dolssa                                                    |        5 |       28.32 | Historical Fiction |
| 1,000 Places to See Before You Die                                       |        5 |       26.08 | Travel             |
| What Happened on Beale Street (Secrets of the South Mysteries #2)        |        5 |       25.37 | Mystery            |
| The Silkworm (Cormoran Strike #2)                                        |        5 |       23.05 | Mystery            |

Equivalent output: **True**
