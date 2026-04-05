import mysql.connector
from mysql.connector import Error
import pandas as pd

# ── Connection Settings ───────────────────────────────────────────────────────
config = {
    "host":     "127.0.0.1",
    "port":     3306,
    "user":     "root",
    "password": "!Zhanna7878036",
    "database": "movies",
    "raise_on_warnings": True
}

try:
    connection = mysql.connector.connect(**config)

    if connection.is_connected():
        # Use server_info property instead of deprecated get_server_info()
        print(f"✓ Connected to MySQL Server version: {connection.server_info}")
        print(f"✓ Database: {config['database']}\n")

        # ── Use SQLAlchemy to avoid pandas warning ────────────────────────────
        from sqlalchemy import create_engine
        engine = create_engine(
            f"mysql+mysqlconnector://root:!Zhanna7878036@127.0.0.1:3306/movies"
        )

        # ── Query ─────────────────────────────────────────────────────────────
        query = "SELECT * FROM movies_base LIMIT 10;"
        df = pd.read_sql(query, engine)

        # ── Clean display — split into readable sections ──────────────────────
        pd.set_option('display.max_colwidth', 60)
        pd.set_option('display.width', 120)

        print("=" * 60)
        print(f"  MOVIES DATABASE — first {len(df)} rows")
        print(f"  Total columns: {len(df.columns)}")
        print("=" * 60)

        # Section 1: Core info
        print("\n── Core Info ──────────────────────────────────────────")
        core = df[['id', 'title', 'year', 'release_date', 'vote_average', 'popularity']].copy()
        core['title'] = core['title'].str[:40]
        print(core.to_string(index=False))

        # Section 2: Categories
        print("\n── Categories ─────────────────────────────────────────")
        cats = df[['id', 'title', 'genres', 'directors']].copy()
        cats['title']     = cats['title'].str[:25]
        cats['genres']    = cats['genres'].str[:35]
        cats['directors'] = cats['directors'].str[:30]
        print(cats.to_string(index=False))

        # Section 3: Cast
        print("\n── Cast ───────────────────────────────────────────────")
        cast = df[['id', 'title', 'cast']].copy()
        cast['title'] = cast['title'].str[:25]
        cast['cast']  = cast['cast'].str[:60]
        print(cast.to_string(index=False))

        # Section 4: Overview
        print("\n── Overview ───────────────────────────────────────────")
        for _, row in df.iterrows():
            print(f"\n  [{row['id']}] {row['title']} ({row['year']})")
            overview = str(row['overview'])
            # Word-wrap at 70 chars
            words, line = overview.split(), ""
            for word in words:
                if len(line) + len(word) + 1 > 70:
                    print(f"       {line}")
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                print(f"       {line}")

        # Summary stats
        print("\n" + "=" * 60)
        print("  SUMMARY STATS")
        print("=" * 60)
        print(f"  Rows fetched       : {len(df)}")
        print(f"  Avg vote_average   : {df['vote_average'].mean():.3f}")
        print(f"  Avg popularity     : {df['popularity'].mean():.3f}")
        print(f"  Year range         : {df['year'].min()} – {df['year'].max()}")
        print(f"  Unique genres      : {df['genres'].nunique()}")

except Error as e:
    print(f"✗ Connection failed: {e}")
    print("\nCommon fixes:")
    print("  - Check your password is correct")
    print("  - Make sure MySQL80 service is running")
    print("  - Verify the database name exists")

finally:
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("\n✓ Connection closed.")
