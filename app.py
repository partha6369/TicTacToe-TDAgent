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

def get_board_df():
    grid = np.array(board).reshape(3, 3)
    return pd.DataFrame(grid, columns=["A", "B", "C"], index=["1", "2", "3"])

def reset_game():
    global board, game_over
    board = [' '] * 9
    game_over = False
    return get_board_df(), "Your move."

def play_human(square):
    global board, game_over

    try:
        square = int(square)
        if square < 0 or square > 8:
            return get_board_df(), "❌ Invalid move. Enter 0–8."
    except:
        return get_board_df(), "❌ Invalid input. Enter number 0–8."

    if game_over:
        return get_board_df(), "⚠️ Game is over. Click Reset."

    if board[square] != ' ':
        return get_board_df(), "❌ Square already taken. Try another."

    board[square] = human_symbol

    if check_winner(board, human_symbol):
        game_over = True
        td_agent.update(-1)
        return get_board_df(), "🏆 You win!"
    elif is_draw(board):
        game_over = True
        td_agent.update(0)
        return get_board_df(), "🤝 It's a draw."

    # Agent's turn
    td_agent.remember(board.copy())
    agent_move = td_agent.choose_action(board, agent_symbol)
    board[agent_move] = agent_symbol

    if check_winner(board, agent_symbol):
        game_over = True
        td_agent.update(1)
        return get_board_df(), "🤖 Agent wins!"
    elif is_draw(board):
        game_over = True
        td_agent.update(0)
        return get_board_df(), "🤝 It's a draw."

    # Update after neutral move
    td_agent.update(0.5, board)
    return get_board_df(), "Your move."

# Try to load the PayPal URL from the environment; if missing, use a placeholder
paypal_url = os.getenv("PAYPAL_URL", "https://www.paypal.com/donate/dummy-link")

# --- Gradio Interface ---
with gr.Blocks(title="Tic Tac Toe with TD Agent") as demo:
    gr.Markdown("## 🎯 Play Tic Tac Toe Against a TD-Learning Agent")

    board_display = gr.Dataframe(label="Game Board", row_count=3, col_count=3, interactive=False)
    game_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        square_input = gr.Number(label="Your move (0–8)", precision=0)
        move_btn = gr.Button("Play Move")
        reset_btn = gr.Button("Reset Game")

    move_btn.click(fn=play_human, inputs=square_input, outputs=[board_display, game_status])
    reset_btn.click(fn=reset_game, inputs=None, outputs=[board_display, game_status])

    with gr.Row():
        gr.HTML(f"""
        <a href="{paypal_url}" target="_blank">
            <button style="background-color:#0070ba;color:white;border:none;padding:10px 20px;
            font-size:16px;border-radius:5px;cursor:pointer;margin-top:10px;">
                ❤️ Support Research via PayPal
            </button>
        </a>
        """)

    demo.load(fn=reset_game, inputs=None, outputs=[board_display, game_status])

if __name__ == "__main__":
    # Determine if running on Hugging Face Spaces
    on_spaces = os.environ.get("SPACE_ID") is not None

    # Launch the app conditionally
    demo.launch(share=not on_spaces)
