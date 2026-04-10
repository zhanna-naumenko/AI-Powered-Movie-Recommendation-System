import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
import re
import threading
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.vq import kmeans, whiten
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#DB CONFIG
DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     3306,
    "user":     "root",
    "password": "**************",
    "database": "movies",
}

#COLORS
BG       = "#0d0d0d"
SURFACE  = "#161616"
CARD     = "#1e1e1e"
ACCENT   = "#e2b04a"
ACCENT2  = "#4df0b4"
TEXT     = "#f0ede6"
MUTED    = "#666666"
DANGER   = "#e05252"
SUCCESS  = "#4df0b4"
BORDER   = "#2a2a2a"

FONT_H1  = ("Georgia", 20, "bold")
FONT_H2  = ("Georgia", 14, "bold")
FONT_H3  = ("Georgia", 11, "bold")
FONT_B   = ("Courier New", 10)
FONT_SM  = ("Courier New", 9)

#DATABASE LAYER
class DB:
    @staticmethod
    def conn():
        return mysql.connector.connect(**DB_CONFIG)

    @staticmethod
    def engine():
        c = DB_CONFIG
        return create_engine(
            f"mysql+mysqlconnector://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['database']}"
        )

    @staticmethod
    def get_user(nickname: str):
        try:
            cn = DB.conn()
            cur = cn.cursor(dictionary=True)
            cur.execute("SELECT * FROM users WHERE user_nickname=%s", (nickname,))
            row = cur.fetchone()
            cur.close(); cn.close()
            return row
        except Error:
            return None

    @staticmethod
    def add_user(user_name, user_nickname, age, genre_prefs, watch_history):
        sql = """INSERT INTO users (user_name,user_nickname,age,genre_preferences,watch_history)
                 VALUES (%s,%s,%s,%s,%s)"""
        cn = DB.conn(); cur = cn.cursor()
        cur.execute(sql, (user_name, user_nickname, age,
                          json.dumps(genre_prefs), json.dumps(watch_history)))
        cn.commit(); cur.close(); cn.close()

    @staticmethod
    def add_watch_entry(nickname: str, entry: dict):
        sql = """UPDATE users
                 SET watch_history=JSON_ARRAY_APPEND(watch_history,'$',CAST(%s AS JSON)),
                     updated_at=CURRENT_TIMESTAMP
                 WHERE user_nickname=%s"""
        cn = DB.conn(); cur = cn.cursor()
        cur.execute(sql, (json.dumps(entry), nickname))
        cn.commit(); cur.close(); cn.close()

    @staticmethod
    def update_genres(nickname: str, genres: list):
        sql = "UPDATE users SET genre_preferences=%s, updated_at=CURRENT_TIMESTAMP WHERE user_nickname=%s"
        cn = DB.conn(); cur = cn.cursor()
        cur.execute(sql, (json.dumps(genres), nickname))
        cn.commit(); cur.close(); cn.close()

    @staticmethod
    def all_users():
        try:
            cn = DB.conn(); cur = cn.cursor(dictionary=True)
            cur.execute("SELECT * FROM users")
            rows = cur.fetchall(); cur.close(); cn.close()
            return rows
        except Error:
            return []

    @staticmethod
    def movies_df():
        try:
            eng = DB.engine()
            return pd.read_sql("SELECT id,title,genres,vote_average,popularity FROM movies_base LIMIT 5000", eng)
        except Exception:
            return pd.DataFrame()


#RECOMMENDATION ENGINE
class RecommendEngine:
    ALL_GENRES = ["action","adventure","animation","biography","comedy","crime",
                  "documentary","drama","fantasy","history","horror","mystery",
                  "romance","sci-fi","superhero","thriller","war","western"]

    @staticmethod
    def genre_vector(genres: list) -> np.ndarray:
        v = np.zeros(len(RecommendEngine.ALL_GENRES))
        for i, g in enumerate(RecommendEngine.ALL_GENRES):
            if g in [x.lower() for x in genres]:
                v[i] = 1.0
        return v

    @staticmethod
    def pearson_similar_users(current_nickname: str) -> list:
        """Find users with similar genre vectors via Pearson correlation."""
        users = DB.all_users()
        if len(users) < 2:
            return []
        current = next((u for u in users if u["user_nickname"] == current_nickname), None)
        if not current:
            return []
        cur_prefs = json.loads(current["genre_preferences"]) if isinstance(current["genre_preferences"], str) else current["genre_preferences"]
        cur_vec = RecommendEngine.genre_vector(cur_prefs)
        scored = []
        for u in users:
            if u["user_nickname"] == current_nickname:
                continue
            prefs = json.loads(u["genre_preferences"]) if isinstance(u["genre_preferences"], str) else u["genre_preferences"]
            vec = RecommendEngine.genre_vector(prefs)
            if cur_vec.std() == 0 or vec.std() == 0:
                continue
            r, _ = stats.pearsonr(cur_vec, vec)
            scored.append((u["user_nickname"], round(r, 3)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:5]

    @staticmethod
    def kmeans_cluster(current_nickname: str) -> dict:
        """K-means cluster users by genre vector, return cluster mates."""
        users = DB.all_users()
        if len(users) < 3:
            return {"cluster": 0, "members": []}
        nicknames, vectors = [], []
        for u in users:
            prefs = json.loads(u["genre_preferences"]) if isinstance(u["genre_preferences"], str) else u["genre_preferences"]
            nicknames.append(u["user_nickname"])
            vectors.append(RecommendEngine.genre_vector(prefs))
        X = np.array(vectors, dtype=float)
        try:
            Xw = whiten(X + 1e-6)
            k = min(3, len(users))
            centroids, _ = kmeans(Xw, k)
            dists = np.linalg.norm(Xw[:, None] - centroids[None, :], axis=2)
            labels = dists.argmin(axis=1)
            idx = nicknames.index(current_nickname)
            my_cluster = labels[idx]
            members = [nicknames[i] for i in range(len(nicknames))
                       if labels[i] == my_cluster and nicknames[i] != current_nickname]
            return {"cluster": int(my_cluster), "members": members}
        except Exception:
            return {"cluster": 0, "members": []}

    @staticmethod
    def recommend(user: dict, n=8) -> list:
        """Return top-N recommended movies with reason strings."""
        prefs = json.loads(user["genre_preferences"]) if isinstance(user["genre_preferences"], str) else user["genre_preferences"]
        history = json.loads(user["watch_history"]) if isinstance(user["watch_history"], str) else user["watch_history"]
        watched_titles = {e["movie"].lower() for e in history}
        liked = [e["movie"] for e in history if e.get("rating", 0) >= 8]
        df = DB.movies_df()
        if df.empty:
            return []

        results = []
        pref_lower = [p.lower() for p in prefs]

        for _, row in df.iterrows():
            title = str(row["title"])
            if title.lower() in watched_titles:
                continue
            movie_genres = [g.strip().lower() for g in str(row.get("genres","")).split(",")]
            overlap = [g for g in pref_lower if g in movie_genres]
            if not overlap:
                continue
            score = len(overlap) * 10 + float(row.get("vote_average", 0)) + float(row.get("popularity", 0)) / 100

            # Build reason string
            if liked:
                reason = f'Since you liked "{liked[0]}", you might enjoy this {overlap[0]} film.'
            else:
                reason = f'Matches your interest in {", ".join(overlap[:2])}.'
            results.append({
                "title": title,
                "genres": str(row.get("genres","")),
                "score": round(score, 2),
                "vote_average": round(float(row.get("vote_average", 0)), 1),
                "reason": reason
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:n]

#STYLED WIDGETS
def styled_btn(parent, text, cmd, color=ACCENT, fg="#000", width=22):
    btn = tk.Button(parent, text=text, command=cmd,
                    bg=color, fg=fg, font=("Courier New", 10, "bold"),
                    relief="flat", cursor="hand2", width=width,
                    activebackground=color, activeforeground=fg,
                    bd=0, padx=10, pady=8)
    btn.bind("<Enter>", lambda e: btn.config(bg=_lighten(color)))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn

def _lighten(hex_color):
    try:
        r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
        r = min(255, r+30); g = min(255, g+30); b = min(255, b+30)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color

def label(parent, text, font=FONT_B, fg=TEXT, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg, bg=BG, **kw)

def card_frame(parent, **kw):
    return tk.Frame(parent, bg=CARD, bd=0, highlightbackground=BORDER,
                    highlightthickness=1, **kw)

#SCREENS
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎬 CineAI — Movie Recommendation System")
        self.geometry("860x680")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.current_user = None
        self._frame = None
        self.show_login()

    def switch(self, frame_class, *args, **kwargs):
        if self._frame:
            self._frame.destroy()
        self._frame = frame_class(self, *args, **kwargs)
        self._frame.pack(fill="both", expand=True)

    def show_login(self):    self.switch(LoginScreen)
    def show_menu(self):     self.switch(MenuScreen, self.current_user)
    def show_register(self): self.switch(RegisterScreen)


#LOGIN
class LoginScreen(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self._build()

    def _build(self):
        tk.Label(self, text="🎬", font=("Georgia", 48), bg=BG, fg=ACCENT).pack(pady=(50,4))
        tk.Label(self, text="CineAI", font=("Georgia", 28, "bold"), bg=BG, fg=TEXT).pack()
        tk.Label(self, text="AI-Powered Movie Recommendations", font=FONT_SM, bg=BG, fg=MUTED).pack(pady=(2,30))

        frm = card_frame(self, padx=30, pady=28)
        frm.pack(ipadx=10, ipady=10)

        tk.Label(frm, text="Enter your nickname", font=FONT_H3, bg=CARD, fg=MUTED).pack()
        self.entry = tk.Entry(frm, font=("Courier New", 13), bg=SURFACE, fg=TEXT,
                              insertbackground=ACCENT, relief="flat", width=24,
                              highlightbackground=BORDER, highlightthickness=1)
        self.entry.pack(pady=10, ipady=6)
        self.entry.bind("<Return>", lambda e: self._login())

        styled_btn(frm, "  Enter  →", self._login, ACCENT, "#000").pack(pady=4)
        tk.Label(frm, text="─" * 30, bg=CARD, fg=BORDER).pack(pady=6)
        styled_btn(frm, "  New user? Register", self.master.show_register, SURFACE, ACCENT2).pack()

        self.status = tk.Label(self, text="", font=FONT_SM, bg=BG, fg=DANGER)
        self.status.pack(pady=8)

    def _login(self):
        nick = self.entry.get().strip()
        if not nick:
            self.status.config(text="Please enter a nickname.")
            return
        self.status.config(text="Connecting...", fg=MUTED)
        self.update()
        user = DB.get_user(nick)
        if user:
            self.master.current_user = user
            self.master.show_menu()
        else:
            self.status.config(text=f"Nickname '{nick}' not found. Please register.", fg=DANGER)


#REGISTER
class RegisterScreen(tk.Frame):
    ALL_GENRES = RecommendEngine.ALL_GENRES

    def __init__(self, master):
        super().__init__(master, bg=BG)
        self._genre_vars = {}
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=24, pady=(18,0))
        styled_btn(top, "← Back", self.master.show_login, SURFACE, ACCENT, width=10).pack(side="left")
        tk.Label(top, text="Create Account", font=FONT_H1, bg=BG, fg=TEXT).pack(side="left", padx=16)

        canvas = tk.Canvas(self, bg=BG, bd=0, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))

        frm = card_frame(inner, padx=28, pady=20)
        frm.pack(fill="x", padx=40, pady=16)

        fields = [("Full Name", "name"), ("Nickname", "nick"), ("Age", "age")]
        self._vars = {}
        for label_text, key in fields:
            tk.Label(frm, text=label_text, font=FONT_SM, bg=CARD, fg=MUTED, anchor="w").pack(fill="x")
            e = tk.Entry(frm, font=FONT_B, bg=SURFACE, fg=TEXT, insertbackground=ACCENT,
                         relief="flat", highlightbackground=BORDER, highlightthickness=1)
            e.pack(fill="x", pady=(2,10), ipady=5)
            self._vars[key] = e

        tk.Label(frm, text="Genre Preferences (select all that apply)",
                 font=FONT_SM, bg=CARD, fg=MUTED, anchor="w").pack(fill="x", pady=(4,6))
        gf = tk.Frame(frm, bg=CARD)
        gf.pack(fill="x")
        for i, g in enumerate(self.ALL_GENRES):
            var = tk.BooleanVar()
            cb = tk.Checkbutton(gf, text=g.capitalize(), variable=var,
                                bg=CARD, fg=TEXT, selectcolor=SURFACE,
                                activebackground=CARD, font=FONT_SM)
            cb.grid(row=i//4, column=i%4, sticky="w", padx=6, pady=2)
            self._genre_vars[g] = var

        self.status = tk.Label(frm, text="", font=FONT_SM, bg=CARD, fg=DANGER)
        self.status.pack(pady=4)
        styled_btn(frm, "  Create Account  →", self._register, ACCENT2, "#000", width=26).pack(pady=8)

    def _register(self):
        name = self._vars["name"].get().strip()
        nick = self._vars["nick"].get().strip()
        age_s = self._vars["age"].get().strip()
        if not all([name, nick, age_s]):
            self.status.config(text="All fields are required.", fg=DANGER); return
        if not age_s.isdigit():
            self.status.config(text="Age must be a number.", fg=DANGER); return
        age = int(age_s)
        genres = [g for g, v in self._genre_vars.items() if v.get()]
        if not genres:
            self.status.config(text="Select at least one genre.", fg=DANGER); return
        if DB.get_user(nick):
            self.status.config(text=f"Nickname '{nick}' already taken.", fg=DANGER); return
        try:
            DB.add_user(name, nick, age, genres, [])
            self.master.current_user = DB.get_user(nick)
            self.master.show_menu()
        except Exception as ex:
            self.status.config(text=f"Error: {ex}", fg=DANGER)


#MAIN MENU
class MenuScreen(tk.Frame):
    def __init__(self, master, user):
        super().__init__(master, bg=BG)
        self.user = user
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=SURFACE, pady=16)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎬  CineAI", font=FONT_H1, bg=SURFACE, fg=ACCENT).pack()
        prefs = json.loads(self.user["genre_preferences"]) if isinstance(self.user["genre_preferences"], str) else self.user["genre_preferences"]
        tk.Label(hdr, text=f"Welcome back, {self.user['user_name']}  ·  {', '.join(prefs[:3])}",
                 font=FONT_SM, bg=SURFACE, fg=MUTED).pack()

        # Menu grid
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=40, pady=30)

        items = [
            ("🎥", "Get Recommendations",  "AI-powered picks for you",        self._recommendations, ACCENT),
            ("➕", "Add Watch History",     "Log a movie you just watched",     self._add_watch,       ACCENT2),
            ("🎭", "Update Genres",         "Refresh your taste profile",       self._update_genres,   "#7eb3f5"),
            ("📊", "Statistics",            "Visualise your watch history",     self._statistics,      "#f5a742"),
            ("👥", "Similar Users",         "Find users who think like you",    self._similar_users,   "#c47af5"),
            ("🚪", "Log Out",               "Switch account",                   self.master.show_login, MUTED),
        ]

        for i, (icon, title, sub, cmd, color) in enumerate(items):
            c = card_frame(body, padx=20, pady=16, cursor="hand2")
            c.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")
            c.bind("<Button-1>", lambda e, f=cmd: f())
            tk.Label(c, text=icon, font=("Georgia", 26), bg=CARD, fg=color).pack(anchor="w")
            tk.Label(c, text=title, font=FONT_H3, bg=CARD, fg=TEXT).pack(anchor="w")
            tk.Label(c, text=sub,   font=FONT_SM,  bg=CARD, fg=MUTED).pack(anchor="w", pady=(2,0))
            styled_btn(c, "Open →", cmd, SURFACE, color, width=12).pack(anchor="e", pady=(8,0))

        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

    def _refresh_user(self):
        self.user = DB.get_user(self.user["user_nickname"])

    #Recommendations
    def _recommendations(self):
        self._refresh_user()
        win = _popup("AI Recommendations", 700, 540)
        tk.Label(win, text="🎬  Your Picks", font=FONT_H2, bg=BG, fg=ACCENT).pack(pady=(16,4))
        tk.Label(win, text="Powered by genre overlap + Pearson correlation + K-Means", font=FONT_SM, bg=BG, fg=MUTED).pack()

        # Similar users (Pearson)
        similar = RecommendEngine.pearson_similar_users(self.user["user_nickname"])
        # K-means cluster
        cluster = RecommendEngine.kmeans_cluster(self.user["user_nickname"])

        info = tk.Frame(win, bg=BG)
        info.pack(fill="x", padx=20, pady=8)
        if similar:
            tk.Label(info, text=f"👥 Most similar user: {similar[0][0]} (r={similar[0][1]})",
                     font=FONT_SM, bg=BG, fg=ACCENT2).pack(anchor="w")
        if cluster["members"]:
            tk.Label(info, text=f"📦 Your cluster #{cluster['cluster']} mates: {', '.join(cluster['members'][:3])}",
                     font=FONT_SM, bg=BG, fg="#7eb3f5").pack(anchor="w")

        tk.Label(win, text="─"*80, bg=BG, fg=BORDER).pack()

        scroll_frame = _scrollable(win)
        recs = RecommendEngine.recommend(self.user)
        if not recs:
            tk.Label(scroll_frame, text="No recommendations found. Update your genre preferences.",
                     font=FONT_B, bg=BG, fg=MUTED).pack(pady=20)
        for r in recs:
            c = card_frame(scroll_frame, padx=14, pady=10)
            c.pack(fill="x", padx=16, pady=5)
            top = tk.Frame(c, bg=CARD)
            top.pack(fill="x")
            tk.Label(top, text=r["title"], font=FONT_H3, bg=CARD, fg=TEXT).pack(side="left")
            tk.Label(top, text=f"★ {r['vote_average']}/10", font=FONT_SM, bg=CARD, fg=ACCENT).pack(side="right")
            tk.Label(c, text=r["genres"], font=FONT_SM, bg=CARD, fg=MUTED).pack(anchor="w")
            tk.Label(c, text=f"💡 {r['reason']}", font=FONT_SM, bg=CARD, fg=ACCENT2, wraplength=560, justify="left").pack(anchor="w", pady=(4,0))

    #Add Watch History
    def _add_watch(self):
        win = _popup("Add Watch History", 420, 320)
        tk.Label(win, text="➕  Log a Movie", font=FONT_H2, bg=BG, fg=ACCENT2).pack(pady=(16,12))
        frm = card_frame(win, padx=24, pady=18)
        frm.pack(padx=24)

        fields = {}
        for lbl, key in [("Movie Title", "movie"), ("Genre", "genre"), ("Rating (1-10)", "rating")]:
            tk.Label(frm, text=lbl, font=FONT_SM, bg=CARD, fg=MUTED, anchor="w").pack(fill="x")
            e = tk.Entry(frm, font=FONT_B, bg=SURFACE, fg=TEXT, insertbackground=ACCENT,
                         relief="flat", highlightbackground=BORDER, highlightthickness=1)
            e.pack(fill="x", pady=(2,8), ipady=4)
            fields[key] = e

        status = tk.Label(frm, text="", font=FONT_SM, bg=CARD, fg=DANGER)
        status.pack()

        def submit():
            movie = fields["movie"].get().strip()
            genre = fields["genre"].get().strip()
            rating_s = fields["rating"].get().strip()
            if not all([movie, genre, rating_s]):
                status.config(text="All fields required."); return
            if not rating_s.isdigit() or not (1 <= int(rating_s) <= 10):
                status.config(text="Rating must be 1–10."); return
            try:
                DB.add_watch_entry(self.user["user_nickname"],
                                   {"movie": movie, "genre": genre, "rating": int(rating_s)})
                status.config(text=f"✓ '{movie}' added!", fg=SUCCESS)
                self._refresh_user()
                win.after(1200, win.destroy)
            except Exception as ex:
                status.config(text=f"Error: {ex}", fg=DANGER)

        styled_btn(frm, "Save →", submit, ACCENT2, "#000", width=20).pack(pady=6)

    #Update Genres
    def _update_genres(self):
        win = _popup("Update Genre Preferences", 460, 360)
        tk.Label(win, text="🎭  Update Genres", font=FONT_H2, bg=BG, fg="#7eb3f5").pack(pady=(16,8))
        current = json.loads(self.user["genre_preferences"]) if isinstance(self.user["genre_preferences"], str) else self.user["genre_preferences"]

        frm = card_frame(win, padx=20, pady=16)
        frm.pack(padx=24, fill="x")
        gvars = {}
        for i, g in enumerate(RecommendEngine.ALL_GENRES):
            var = tk.BooleanVar(value=(g in current))
            cb = tk.Checkbutton(frm, text=g.capitalize(), variable=var,
                                bg=CARD, fg=TEXT, selectcolor=SURFACE,
                                activebackground=CARD, font=FONT_SM)
            cb.grid(row=i//4, column=i%4, sticky="w", padx=6, pady=2)
            gvars[g] = var

        status = tk.Label(win, text="", font=FONT_SM, bg=BG, fg=DANGER)
        status.pack(pady=4)

        def save():
            selected = [g for g, v in gvars.items() if v.get()]
            if not selected:
                status.config(text="Select at least one genre.", fg=DANGER); return
            DB.update_genres(self.user["user_nickname"], selected)
            self._refresh_user()
            status.config(text="✓ Preferences saved!", fg=SUCCESS)
            win.after(1000, win.destroy)

        styled_btn(win, "Save Preferences →", save, "#7eb3f5", "#000", width=24).pack(pady=8)

    #Statistics
    def _statistics(self):
        self._refresh_user()
        history = json.loads(self.user["watch_history"]) if isinstance(self.user["watch_history"], str) else self.user["watch_history"]
        if not history:
            messagebox.showinfo("Statistics", "No watch history yet. Add some movies first!"); return

        win = _popup("Statistics", 700, 580)
        tk.Label(win, text="📊  Your Statistics", font=FONT_H2, bg=BG, fg="#f5a742").pack(pady=(14,2))
        tk.Label(win, text=f"{len(history)} movies watched", font=FONT_SM, bg=BG, fg=MUTED).pack()

        # Build data
        genre_counts = {}
        ratings = []
        for e in history:
            g = e.get("genre", "unknown").strip().lower()
            genre_counts[g] = genre_counts.get(g, 0) + 1
            try:
                ratings.append(float(e.get("rating", 0)))
            except Exception:
                pass

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0d0d0d")
        colors = ["#e2b04a","#4df0b4","#7eb3f5","#f5a742","#c47af5","#e05252","#5cdbe0","#f0ede6"]

        # Pie — genre distribution
        ax1 = axes[0]
        ax1.set_facecolor("#0d0d0d")
        labels_list = list(genre_counts.keys())
        sizes = list(genre_counts.values())
        wedge_colors = colors[:len(labels_list)]
        ax1.pie(sizes, labels=labels_list, colors=wedge_colors,
                autopct="%1.0f%%", textprops={"color": "#f0ede6", "fontsize": 9},
                wedgeprops={"edgecolor": "#0d0d0d", "linewidth": 1.5})
        ax1.set_title("Genre Distribution", color="#f0ede6", fontsize=11, pad=10)

        # Bar — ratings
        ax2 = axes[1]
        ax2.set_facecolor("#161616")
        ax2.spines[:].set_color("#2a2a2a")
        ax2.tick_params(colors="#666666")
        if ratings:
            bins = range(1, 12)
            ax2.hist(ratings, bins=bins, color="#e2b04a", edgecolor="#0d0d0d", rwidth=0.8, align="left")
            ax2.axvline(np.mean(ratings), color="#4df0b4", linewidth=1.5, linestyle="--",
                        label=f"avg {np.mean(ratings):.1f}")
            ax2.legend(fontsize=8, labelcolor="#4df0b4", facecolor="#161616", edgecolor="#2a2a2a")
        ax2.set_title("Rating Distribution", color="#f0ede6", fontsize=11)
        ax2.set_xlabel("Rating (1–10)", color="#666666", fontsize=9)
        ax2.set_ylabel("Count", color="#666666", fontsize=9)
        ax2.set_xticks(range(1, 11))

        plt.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=8)

        # Summary stats
        sf = card_frame(win, padx=16, pady=10)
        sf.pack(fill="x", padx=16, pady=(0,12))
        if ratings:
            stats_text = (f"  Total movies: {len(history)}    "
                          f"Avg rating: {np.mean(ratings):.1f}/10    "
                          f"Highest: {max(ratings):.0f}    "
                          f"Lowest: {min(ratings):.0f}    "
                          f"Favourite genre: {max(genre_counts, key=genre_counts.get).capitalize()}")
            tk.Label(sf, text=stats_text, font=FONT_SM, bg=CARD, fg=ACCENT).pack()

    #Similar Users
    def _similar_users(self):
        win = _popup("Similar Users", 500, 420)
        tk.Label(win, text="👥  Users Like You", font=FONT_H2, bg=BG, fg="#c47af5").pack(pady=(16,4))
        tk.Label(win, text="Based on Pearson correlation of genre preferences", font=FONT_SM, bg=BG, fg=MUTED).pack()

        similar = RecommendEngine.pearson_similar_users(self.user["user_nickname"])
        cluster = RecommendEngine.kmeans_cluster(self.user["user_nickname"])

        tk.Label(win, text=f"\n📦 K-Means Cluster: #{cluster['cluster']}", font=FONT_H3, bg=BG, fg="#7eb3f5").pack()
        if cluster["members"]:
            tk.Label(win, text="Cluster mates: " + ", ".join(cluster["members"]),
                     font=FONT_SM, bg=BG, fg=TEXT, wraplength=440).pack(pady=4)
        else:
            tk.Label(win, text="No cluster mates yet (need more users).", font=FONT_SM, bg=BG, fg=MUTED).pack()

        tk.Label(win, text="\n📈 Pearson Correlation Rankings", font=FONT_H3, bg=BG, fg=ACCENT2).pack()
        sf = _scrollable(win)
        if not similar:
            tk.Label(sf, text="Not enough users for correlation.", font=FONT_B, bg=BG, fg=MUTED).pack(pady=20)
        for nick, score in similar:
            u = DB.get_user(nick)
            c = card_frame(sf, padx=14, pady=8)
            c.pack(fill="x", padx=16, pady=4)
            top = tk.Frame(c, bg=CARD); top.pack(fill="x")
            tk.Label(top, text=f"@{nick}", font=FONT_H3, bg=CARD, fg=TEXT).pack(side="left")
            color = ACCENT2 if score > 0.5 else (ACCENT if score > 0 else DANGER)
            tk.Label(top, text=f"r = {score}", font=FONT_SM, bg=CARD, fg=color).pack(side="right")
            if u:
                prefs = json.loads(u["genre_preferences"]) if isinstance(u["genre_preferences"], str) else u["genre_preferences"]
                tk.Label(c, text=", ".join(prefs[:5]), font=FONT_SM, bg=CARD, fg=MUTED).pack(anchor="w")

#HELPERS
def _popup(title: str, w: int, h: int) -> tk.Toplevel:
    win = tk.Toplevel()
    win.title(title)
    win.geometry(f"{w}x{h}")
    win.configure(bg=BG)
    win.resizable(True, True)
    return win

def _scrollable(parent) -> tk.Frame:
    canvas = tk.Canvas(parent, bg=BG, bd=0, highlightthickness=0)
    sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    canvas.pack(fill="both", expand=True)
    inner = tk.Frame(canvas, bg=BG)
    win_id = canvas.create_window((0,0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
    return inner


# --------------------------------------------------------------------------------------------------------
#ENTRY POINT
if __name__ == "__main__":
    app = App()
    app.mainloop()
