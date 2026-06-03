from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
import psycopg2
import os

# Load .env
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def home():

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        query = """
        SELECT
            cmetadata->>'class_name' AS class,
            cmetadata->>'subject' AS subject,
            cmetadata->>'chapter_name' AS chapter,
            COUNT(*) AS chunks
        FROM langchain_pg_embedding
        GROUP BY class, subject, chapter
        ORDER BY class, subject, chapter;
        """

        cursor.execute(query)

        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        table_rows = ""

        for row in rows:
            table_rows += f"""
            <tr>
                <td>{row[0]}</td>
                <td>{row[1]}</td>
                <td>{row[2]}</td>
                <td>{row[3]}</td>
            </tr>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Chapter Viewer</title>

            <style>

                body {{
                    font-family: Arial;
                    background: #f5f5f5;
                    padding: 30px;
                }}

                h1 {{
                    text-align: center;
                    margin-bottom: 30px;
                }}

                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}

                th {{
                    background: #111827;
                    color: white;
                    padding: 14px;
                    text-align: left;
                }}

                td {{
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                }}

                tr:hover {{
                    background: #f3f4f6;
                }}

            </style>

        </head>

        <body>

            <h1>📚 Classes & Chapters Viewer</h1>

            <table>

                <tr>
                    <th>Class</th>
                    <th>Subject</th>
                    <th>Chapter</th>
                    <th>Total Chunks</th>
                </tr>

                {table_rows}

            </table>

        </body>
        </html>
        """

        return HTMLResponse(content=html)

    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error</h1><p>{str(e)}</p>",
            status_code=500
        )