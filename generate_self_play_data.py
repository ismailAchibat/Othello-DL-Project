import argparse
import itertools
import os
import uuid
import h5py
import numpy as np
import torch
import copy
from tqdm import tqdm

from utile import get_legal_moves, initialze_board, has_tile_to_flip, BOARD_SIZE

# --- Functions copied from game.py to avoid running its main loop on import ---

def input_seq_generator(board_stats_seq, length_seq):
    """Prepares the sequence of board states for model input."""
    board_stat_init = initialze_board()
    if len(board_stats_seq) >= length_seq:
        input_seq = board_stats_seq[-length_seq:]
    else:
        # Pad with initial board state if history is not long enough
        input_seq = [board_stat_init] * (length_seq - len(board_stats_seq))
        input_seq.extend(board_stats_seq)
    return input_seq

def find_best_move(move_prob, legal_moves):
    """Finds the best legal move given the model's output probabilities."""
    best_move = legal_moves[0]
    max_score = move_prob[legal_moves[0][0], legal_moves[0][1]]
    for move in legal_moves:
        if move_prob[move[0], move[1]] > max_score:
            max_score = move_prob[move[0], move[1]]
            best_move = move
    return best_move

def apply_flip(best_move, board_stat, NgBlackPsWhith):
    """Apply tile flipping on the Othello board based on the best move."""
    MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1),
                 (0, -1),           (0, +1),
                 (+1, -1), (+1, 0), (+1, +1)]

    for direction in MOVE_DIRS:
        if has_tile_to_flip(best_move, direction, board_stat, NgBlackPsWhith):
            i = 1
            while True:
                row = best_move[0] + direction[0] * i
                col = best_move[1] + direction[1] * i
                if board_stat[row][col] == board_stat[best_move[0], best_move[1]]:
                    break
                else:
                    board_stat[row][col] = board_stat[best_move[0], best_move[1]]
                    i += 1
    return board_stat

# --- New functions for data generation ---

def save_game_to_hdf5(game_data, output_dir, game_id):
    """Saves the game log to an HDF5 file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, game_id)
        with h5py.File(file_path, 'w') as h5f:
            h5f.create_dataset(game_id.replace('.h5', ''), data=game_data, compression="gzip")
    except Exception as e:
        print(f"Error saving HDF5 file {game_id}: {e}")

def play_one_game(player1_path, player2_path, device):
    """
    Simulates a single game between two models, with player1 as Black.
    Returns a tuple of (list_of_board_states, list_of_move_matrices).
    """
    # Load models
    try:
        p1_model = torch.load(player1_path, map_location=device, weights_only=False)
        p1_model.eval()
        p2_model = torch.load(player2_path, map_location=device, weights_only=False)
        p2_model.eval()
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

    board_states_log = []
    move_matrices_log = []
    
    board_stat = initialze_board()
    game_history_seq = []
    pass_counter = 0

    while True:
        # --- Player 1 (Black, -1) ---
        current_player_id = -1
        game_history_seq.append(copy.copy(board_stat))
        legal_moves = get_legal_moves(board_stat, current_player_id)

        if not legal_moves:
            pass_counter += 1
        else:
            pass_counter = 0
            # Prepare input for the model
            input_seq = input_seq_generator(game_history_seq, p1_model.len_inpout_seq)
            model_input = np.array(input_seq) * current_player_id # Normalize for current player
            
            with torch.no_grad():
                move_prob = p1_model(torch.tensor(model_input).float().to(device))
                move_prob = move_prob.cpu().detach().numpy().reshape(8, 8)

            best_move = find_best_move(move_prob, legal_moves)
            
            # Log the state and the move
            board_states_log.append(copy.copy(board_stat))
            move_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
            move_matrix[best_move[0], best_move[1]] = 1
            move_matrices_log.append(move_matrix)

            # Apply the move
            board_stat[best_move[0], best_move[1]] = current_player_id
            board_stat = apply_flip(best_move, board_stat, current_player_id)

        if pass_counter >= 2 or np.all(board_stat):
            break

        # --- Player 2 (White, 1) ---
        current_player_id = 1
        game_history_seq.append(copy.copy(board_stat))
        legal_moves = get_legal_moves(board_stat, current_player_id)

        if not legal_moves:
            pass_counter += 1
        else:
            pass_counter = 0
            input_seq = input_seq_generator(game_history_seq, p2_model.len_inpout_seq)
            model_input = np.array(input_seq) # No normalization needed for white player (id=1)
            
            with torch.no_grad():
                move_prob = p2_model(torch.tensor(model_input).float().to(device))
                move_prob = move_prob.cpu().detach().numpy().reshape(8, 8)

            best_move = find_best_move(move_prob, legal_moves)

            board_states_log.append(copy.copy(board_stat))
            move_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
            move_matrix[best_move[0], best_move[1]] = 1
            move_matrices_log.append(move_matrix)
            
            board_stat[best_move[0], best_move[1]] = current_player_id
            board_stat = apply_flip(best_move, board_stat, current_player_id)

        if pass_counter >= 2 or np.all(board_stat):
            break
            
    return board_states_log, move_matrices_log


def main(args):
    """Main orchestration function."""
    if not args.models or len(args.models) < 2:
        print("Error: Please provide at least two models using the --models argument.")
        return

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Running on device: {device}")

    # Create all unique pairings for the tournament
    model_pairs = list(itertools.combinations(args.models, 2))
    total_games = len(model_pairs) * 2 # Each pair plays twice
    
    new_game_files = []
    
    with tqdm(total=total_games, desc="Generating Self-Play Data") as pbar:
        for model_a_path, model_b_path in model_pairs:
            # Game 1: A (Black) vs B (White)
            pbar.set_description(f"{os.path.basename(model_a_path)} vs {os.path.basename(model_b_path)}")
            boards1, moves1 = play_one_game(model_a_path, model_b_path, device)
            if boards1:
                game_id1 = f"selfplay_{uuid.uuid4().hex[:12]}.h5"
                save_game_to_hdf5([boards1, moves1], args.output_dir, game_id1)
                new_game_files.append(game_id1)
            pbar.update(1)

            # Game 2: B (Black) vs A (White)
            pbar.set_description(f"{os.path.basename(model_b_path)} vs {os.path.basename(model_a_path)}")
            boards2, moves2 = play_one_game(model_b_path, model_a_path, device)
            if boards2:
                game_id2 = f"selfplay_{uuid.uuid4().hex[:12]}.h5"
                save_game_to_hdf5([boards2, moves2], args.output_dir, game_id2)
                new_game_files.append(game_id2)
            pbar.update(1)

    print(f"\nGenerated {len(new_game_files)} new game logs in '{args.output_dir}'.")

    # Append new file names to train.txt
    if new_game_files and args.update_train_txt:
        try:
            with open(args.train_txt_path, 'a') as f:
                for fname in new_game_files:
                    f.write(f"{fname}\n")
            print(f"Successfully appended {len(new_game_files)} new game file names to '{args.train_txt_path}'.")
        except Exception as e:
            print(f"Error updating {args.train_txt_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate self-play data by pitting models against each other.")
    parser.add_argument(
        '--models', 
        nargs='+', 
        required=True, 
        help='A list of paths to the model .pt files for the tournament.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='../dataset/', 
        help="Directory to save the new .h5 game log files. It will be created if it doesn't exist."
    )
    parser.add_argument(
        '--train_txt_path', 
        type=str, 
        default='train.txt',
        help="Path to the train.txt file to append the new game logs to."
    )
    parser.add_argument(
        '--no-update-train-txt',
        dest='update_train_txt',
        action='store_false',
        help="Flag to disable appending the new game file names to train.txt."
    )
    
    args = parser.parse_args()
    main(args)
