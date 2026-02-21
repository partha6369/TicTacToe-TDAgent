# =============================================================================
# Tic-Tac-Toe
# A Temporal Difference (TD) Learning Agent
# 
# An interactive AI-powered Tic-Tac-Toe application where a human player competes
# against a reinforcement learning (Temporal Difference) agent that learns optimal
# strategies through state-value updates.
#
# Copyright © 2026 Dr Partha Majumdar. All Rights Reserved.
#
# This software, including its source code, algorithms, architecture,
# learning logic, and interface design, is the intellectual property of
# Dr Partha Majumdar. Unauthorised copying, modification, distribution,
# reverse engineering, or commercial deployment is strictly prohibited
# without prior written permission.
#
# For licensing or commercial enquiries:
# Contact: partha.majumdar@majumdarconsultancy.in
# =============================================================================

# === Install required libraries ===
import subprocess
import sys

def install_if_missing(pip_package, import_name=None):
    import_name = import_name or pip_package
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", pip_package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


# Attempt to install only if not present
install_if_missing("gradio")
install_if_missing("pandas")
install_if_missing("numpy")

import os
import gradio as gr
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Game tracking
player_counter = 0
stats = defaultdict(lambda: {"matches": 0, "agent_wins": 0})
# current_player = "Anonymous"

def get_today():
    return datetime.now().strftime("Day %j")  # Day number of year

def format_stats_html():
    lines = []
    for player, data in stats.items():
        lines.append(f"{player}: {data['matches']} matches – Agent Wins = {data['agent_wins']}")
    return "<div id='stats-box' style='max-height: 8em; overflow-y: auto; font-family: monospace; white-space: pre-wrap; background: #f8f8f8; padding: 10px; border: 1px solid #ccc;'>" + "<br>".join(lines) + "</div>"

def init_player():
    global player_counter
    player_counter += 1
    player_id = f"Player {player_counter}"
    stats[player_id]["matches"] = 0
    stats[player_id]["agent_wins"] = 0
    return player_id

def init_and_start_game():
    player_id = init_player()
    df, status, stats_text, _ = reset_game(player_id)
    return df, status, stats_text, player_id
    
# --- TD Agent Setup ---
class TDAgent:
    def __init__(self, epsilon=0.1, alpha=0.5):
        self.values = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.last_state = None
        self.last_value = 0

    def get_state_key(self, board):
        return ''.join(board)

    def get_value(self, board):
        key = self.get_state_key(board)
        return self.values.get(key, 0.5)

    def update(self, reward, new_board=None):
        if self.last_state is None:
            return

        prev_key = self.get_state_key(self.last_state)
        prev_value = self.values.get(prev_key, 0.5)

        if new_board is not None:
            next_key = self.get_state_key(new_board)
            next_value = self.values.get(next_key, 0.5)
        else:
            next_value = reward

        self.values[prev_key] = prev_value + self.alpha * (next_value - prev_value)

    def choose_action(self, board, player_symbol):
        if random.random() < self.epsilon:
            empty = [i for i in range(9) if board[i] == ' ']
            return random.choice(empty)

        values = {}
        for i in range(9):
            if board[i] == ' ':
                next_board = board.copy()
                next_board[i] = player_symbol
                values[i] = self.get_value(next_board)

        max_val = max(values.values())
        best_moves = [i for i in values if values[i] == max_val]
        return random.choice(best_moves)

    def remember(self, board):
        self.last_state = board.copy()

# --- Game Logic ---
td_agent = TDAgent()
human_symbol = 'O'
agent_symbol = 'X'
board = [' '] * 9
game_over = False

def check_winner(b, symbol):
    wins = [(0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)]
    return any(b[i] == b[j] == b[k] == symbol for i,j,k in wins)

def is_draw(b):
    return all(cell != ' ' for cell in b)

def get_board_markings():
    board_markings = [str(i) for i in range(9)]
    marking_grid = np.array(board_markings).reshape(3, 3)
    return pd.DataFrame(marking_grid, columns=["A", "B", "C"], index=["1", "2", "3"])

def get_board_df():
    grid = np.array(board).reshape(3, 3)
    return pd.DataFrame(grid, columns=["A", "B", "C"], index=["1", "2", "3"])

def reset_game(player_id):
    global board, game_over
    board = [' '] * 9
    game_over = False

    # Count match
    stats[player_id]["matches"] += 1

    # Agent makes the first move
    agent_move = td_agent.choose_action(board, agent_symbol)
    board[agent_move] = agent_symbol

    # Let agent remember this move for future updates
    td_agent.remember(board.copy())

    return get_board_df(), "Your move.", format_stats_html(), player_id

def play_human(square, player_id):
    global board, game_over

    try:
        square = int(square)
        if square < 0 or square > 8:
            return get_board_df(), "❌ Invalid move. Enter 0–8.", format_stats_html(), player_id
    except:
        return get_board_df(), "❌ Invalid input. Enter number 0–8.", format_stats_html(), player_id

    if game_over:
        return get_board_df(), "⚠️ Game is over. Click Reset.", format_stats_html(), player_id

    if board[square] != ' ':
        return get_board_df(), "❌ Square already taken. Try another.", format_stats_html(), player_id

    board[square] = human_symbol

    if check_winner(board, human_symbol):
        game_over = True
        td_agent.update(-1)
        return get_board_df(), "🏆 You win!", format_stats_html(), player_id
    elif is_draw(board):
        game_over = True
        td_agent.update(0)
        return get_board_df(), "🤝 It's a draw.", format_stats_html(), player_id

    # Agent's turn
    td_agent.remember(board.copy())
    agent_move = td_agent.choose_action(board, agent_symbol)
    board[agent_move] = agent_symbol

    if check_winner(board, agent_symbol):
        game_over = True
        td_agent.update(1)
        stats[player_id]["agent_wins"] += 1  # 👈 Update win stats
        return get_board_df(), "🤖 Agent wins!", format_stats_html(), player_id
    elif is_draw(board):
        game_over = True
        td_agent.update(0)
        return get_board_df(), "🤝 It's a draw.", format_stats_html(), player_id

    # Update after neutral move
    td_agent.update(0.5, board)
    return get_board_df(), "Your move.", format_stats_html(), player_id

# Try to load the PayPal URL from the environment; if missing, use a placeholder
paypal_url = os.getenv("PAYPAL_URL", "https://www.paypal.com/donate/dummy-link")

# --- Gradio Interface ---
static_board_reference = get_board_markings()

APP_TITLE = "🎯 Play Tic Tac Toe Against a TD-Learning Agent"
APP_DESCRIPTION = (
    "Play Tic Tac Toe against a self-learning AI agent powered by Temporal Difference (TD) learning, where each move helps the machine get smarter in real time."
)

with gr.Blocks(title="Tic Tac Toe with TD Agent") as app:
    # Title and Description
    gr.HTML(
        f"""
        <p style='text-align: center; font-size: 40px; font-weight: bold;'>{APP_TITLE}</p>
        <p style='text-align: center; font-size: 20px; color: #555;'><sub>{APP_DESCRIPTION}</sub></p>
        <hr>
        """
    )

    player_state = gr.State()  # Persistent state across interactions

    with gr.Row():
        board_display = gr.Dataframe(label="Game Board", row_count=3, col_count=3, interactive=False)
        gr.Dataframe(label="Game Board Positions", value=static_board_reference, row_count=3, col_count=3, interactive=False)
    
    game_status = gr.Textbox(label="Status", interactive=False)
    stats_display = gr.HTML(value=format_stats_html(), label="Daily Stats")
    player_label = gr.Textbox(label="Player", interactive=False)

    with gr.Row():
        square_input = gr.Number(label="Your move (0–8)", precision=0)
        move_btn = gr.Button("Play Move")
        reset_btn = gr.Button("Reset Game")

    move_btn.click(fn=play_human, inputs=[square_input, player_state], outputs=[board_display, game_status, stats_display, player_label])
    reset_btn.click(fn=reset_game, inputs=player_state, outputs=[board_display, game_status, stats_display, player_label])
    
    app.load(fn=init_and_start_game, inputs=None, outputs=[board_display, game_status, stats_display, player_state])
    
    with gr.Row():
        gr.HTML(f"""
        <a href="{paypal_url}" target="_blank">
            <button style="background-color:#0070ba;color:white;border:none;padding:10px 20px;
            font-size:16px;border-radius:5px;cursor:pointer;margin-top:10px;">
                ❤️ Support Research via PayPal
            </button>
        </a>
        """)

    with gr.Row():
        gr.HTML("""
        <a href="https://huggingface.co/spaces/partha6369/partha-research-centre"
           target="_blank"
           rel="noopener noreferrer">
            <button style="
                background-color:#111827;
                color:white;
                border:none;
                padding:10px 20px;
                font-size:16px;
                border-radius:8px;
                cursor:pointer;
                margin-top:10px;">
                🔗 Dr Partha Majumdar's Research Centre
            </button>
        </a>
        """)

if __name__ == "__main__":
    # Determine if running on Hugging Face Spaces
    on_spaces = os.environ.get("SPACE_ID") is not None

    # Launch the app conditionally
    app.launch(share=not on_spaces)
