import os
import re
import random

DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'


def _check_move_legality(fen: str, uci_move: str) -> float:
    """Returns 1.0 if uci_move is legal on the board described by fen, 0.0 otherwise."""
    if not fen or not uci_move:
        return 0.0
    try:
        import chess
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        return 1.0 if move in board.legal_moves else 0.0
    except Exception:
        return 0.0


if DEBUG:
    print(f"REWARD_MODEL_TYPE: {os.environ.get('REWARD_MODEL_TYPE', 'NOT_SET')}")

REWARD_MODEL_TYPE = os.environ['REWARD_MODEL_TYPE'].upper()


def lan_to_uci(lan: str, side_to_move: str = 'white') -> str:
    """
    Convert custom LAN move to UCI format.
    
    Args:
        lan: Custom LAN string, e.g., "Pd2d4", "Pd4xe5", "Pe7e8=Q", "O-O"
        side_to_move: 'white' or 'black' for castling conversion
        
    Returns:
        UCI string, e.g., "d2d4", "d4e5", "e7e8q", "e1g1"
        
    Raises:
        ValueError if invalid format
    """
    import re
    
    # Strip check/checkmate symbols
    lan = lan.rstrip('+#').strip()
    
    # Handle castling
    if lan == 'O-O':
        if side_to_move == 'white':
            return 'e1g1'
        elif side_to_move == 'black':
            return 'e8g8'
        else:
            raise ValueError("Invalid side_to_move for castling")
    elif lan == 'O-O-O':
        if side_to_move == 'white':
            return 'e1c1'
        elif side_to_move == 'black':
            return 'e8c8'
        else:
            raise ValueError("Invalid side_to_move for castling")
    
    # Pattern for regular moves: Piece from (x)? to (=Promotion)?
    match = re.match(r'^([PNBRQK])([a-h][1-8])(x)?([a-h][1-8])(=([QRBN]))?$', lan)
    if not match:
        raise ValueError(f"Invalid LAN format: {lan}")
    
    piece, from_sq, capture, to_sq, promo_group, promo = match.groups()
    
    uci = from_sq + to_sq
    if promo:
        uci += promo.lower()  # UCI uses lowercase for promotion (q/r/b/n)
    
    return uci

def _is_complete_move(text: str) -> bool:
    """
    Check if text represents a complete chess move in custom format.
    Handles: Pd2d4, Pd4xe5, O-O, Pe7e8=Q, etc., with optional + or #.
    """
    if not text:
        return False
    # Remove trailing check/checkmate symbols
    move = text.rstrip('+#')
    
    # Castling (O-O or O-O-O, possibly with piece K?)
    if move in ['O-O', 'O-O-O']:
        return True
    
    # Pattern: Piece [a-h][1-8] (x)? [a-h][1-8] (= [QRBN])?
    pattern = r'^[PNBRQK][a-h][1-8](x)?[a-h][1-8](=[QRBN])?$'
    if re.match(pattern, move):
        return True
    
    return False


def _extract_first_move(text: str) -> str:
    """
    Extract the first complete move from generated text.
    Handles cases where model generates multiple tokens.
    """
    text = text.strip()
    
    # Split on whitespace
    moves = text.split()
    if not moves:
        return text
    
    for move in moves:
        # Skip move numbers (e.g., "1.", "2.", "1...")
        if re.match(r'^\d+\.{1,3}$', move):
            continue
        if _is_complete_move(move):
            return move
    # If no complete move found, return what we have
    return None


def _extract_move_after_thinking(text: str) -> tuple:
    """
    Extract the move that appears after the </T> token in SFT-generated text.
    
    Strict mode:
    - If </T> exists: parse the first move after </T>
    - If </T> doesn't exist: return None (count as wrong)
    - follows_format is True only if both <T> and </T> are present
    
    Args:
        text: Generated text that may contain <T>...</T> format
        
    Returns:
        tuple: (move_after_T, follows_format)
            - move_after_T: The first complete move found after </T>, or None
            - follows_format: Boolean indicating if text follows <T>...</T> format
    """
    text = text.strip()
    
    # Check if the text follows the <T>...</T> format
    has_closing_T = '</T>' in text
    has_single_closing_T = text.count('</T>') == 1
    follows_format = has_closing_T and has_single_closing_T
    
    if not follows_format:
        # Strict mode: No </T> token found, return None (count as wrong)
        return None, False
    
    # Find the position after </T>
    closing_tag_end = text.find('</T>') + len('</T>')
    text_after_T = text[closing_tag_end:].strip()
    
    if not text_after_T:
        return None, follows_format
    
    # Extract the first complete move after </T>
    first_move = _extract_first_move(text_after_T)
    return first_move, follows_format


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    """
    Compute chess move scores by parsing solution strings and comparing with ground truth.
    Uses the same parsing logic as inference.py to extract chess moves.
    """
    if DEBUG:
        print(f"BATCH INPUT DEBUG: data_sources={len(data_sources)}, solution_strs={len(solution_strs)}, ground_truths={len(ground_truths)}, extra_infos={len(extra_infos)}")
        print(f"REWARD_MODEL_TYPE: {REWARD_MODEL_TYPE}")
    
    results = []
    format_penalty_count = 0
    no_move_extracted_count = 0
    
    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=True
    ):
        import json

        if REWARD_MODEL_TYPE == 'EVAL_ONLY_NONTHINK_BASED':
            # Non-thinking model: extract all moves from the full output (no <T></T> parsing).
            # solution_str contains interleaved player+env moves; player moves are at even indices.
            # Score = fraction of player positions matched against ground truth.
            _gt = ground_truth
            if isinstance(_gt, str):
                try:
                    _gt = json.loads(_gt)
                except json.JSONDecodeError:
                    try:
                        import ast
                        _gt = ast.literal_eval(_gt)
                    except (ValueError, SyntaxError):
                        pass
            if not isinstance(_gt, list):
                _gt = [str(_gt).strip()]

            # Extract all valid moves from output in order
            _extracted = []
            for _tok in solution_str.strip().split():
                if re.match(r'^\d+\.{1,3}$', _tok):
                    continue
                if _is_complete_move(_tok):
                    _extracted.append(_tok.strip())

            # Ground truth is a flat list of SOLVER moves only. In the model's
            # output, solver moves live at even positions of _extracted (odd
            # positions are env replies). So compare _gt[i] to _extracted[2*i].
            _correct = 0
            _total_player = len(_gt)
            for _i in range(_total_player):
                _gt_uci = str(_gt[_i]).strip()
                _pred_idx = 2 * _i
                if _pred_idx < len(_extracted):
                    try:
                        _pred_uci = lan_to_uci(_extracted[_pred_idx])
                    except ValueError:
                        _pred_uci = ''
                    if _pred_uci == _gt_uci:
                        _correct += 1

            if _total_player > 0 and _correct == _total_player:
                score = 1.0
            else:
                score = 0.0
            extracted_move = _extracted[0] if _extracted else None
            try:
                extracted_move_uci = lan_to_uci(extracted_move) if extracted_move else ''
            except ValueError:
                extracted_move_uci = ''

            _first_pred_uci = ''
            if _extracted:
                try:
                    _first_pred_uci = lan_to_uci(_extracted[0])
                except ValueError:
                    pass
            _first_gt_uci = str(_gt[0]).strip() if _gt else ''
            first_move_score = 1.0 if (_first_pred_uci and _first_pred_uci == _first_gt_uci) else 0.0

            _fen = extra_info.get('FEN') or extra_info.get('fen', '') if isinstance(extra_info, dict) else ''
            first_move_legality_score = _check_move_legality(_fen, _first_pred_uci)

            results.append({
                "score": float(score),
                "ground_truth": str(ground_truth),
                "reward_method": "CHESS_MOVE_PARSING",
                "extracted_move": extracted_move if extracted_move else '',
                "extracted_move_uci": extracted_move_uci,
                "format_score": 0.0,
                "first_move_score": first_move_score,
                "first_move_legality_score": first_move_legality_score,
                "data_source": data_source,
            })
            continue

        # Extract move from solution_str (handles both <T></T> format and direct moves)
        extracted_move, follows_format = _extract_move_after_thinking(solution_str)

        # If no </T> tag found, try extracting move directly from the text
        if extracted_move is None and not follows_format:
            extracted_move = _extract_first_move(solution_str)

        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError:
                try:
                    import ast
                    ground_truth = ast.literal_eval(ground_truth)
                except (ValueError, SyntaxError):
                    pass
        
        # Process ground truth (handle list or string format)
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            processed_ground_truth = str(ground_truth[0]).strip()
        else:
            processed_ground_truth = str(ground_truth).strip()
        
        # Normalize moves for comparison (remove spaces, convert to same case if needed)
        extracted_move_uci = ''
        if extracted_move:
            extracted_move = extracted_move.strip()
            processed_ground_truth = processed_ground_truth.strip()
            
            # Compare moves (exact match)
            try:
                extracted_move_uci = lan_to_uci(extracted_move)
            except ValueError:
                extracted_move_uci = ''
            if extracted_move_uci == processed_ground_truth:
                score = 1.0
            else:
                score = 0.0
        else:
            # No valid move extracted - score is 0.0
            score = 0.0
            no_move_extracted_count += 1
            if not follows_format:
                format_penalty_count += 1
        
        if REWARD_MODEL_TYPE == 'RULE_FORMAT_BASED':
            score = score if follows_format else 0.0
        else:
            score = score
        
        first_move_score = 1.0 if (extracted_move_uci and extracted_move_uci == processed_ground_truth) else 0.0

        _fen = extra_info.get('FEN') or extra_info.get('fen', '') if isinstance(extra_info, dict) else ''
        first_move_legality_score = _check_move_legality(_fen, extracted_move_uci)

        results.append({
            "score": float(score),
            "ground_truth": str(ground_truth),
            "reward_method": "CHESS_MOVE_PARSING",
            "extracted_move": extracted_move if extracted_move else '',
            "extracted_move_uci": extracted_move_uci,
            "format_score": 1.0 if follows_format else 0.0,
            "first_move_score": first_move_score,
            "first_move_legality_score": first_move_legality_score,
            "data_source": data_source,
        })
    
    if DEBUG and results:
        correct_count = sum(1 for r in results if r["score"] > 0.5)
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        format_valid_count = total_count - format_penalty_count
        move_extracted_count = total_count - no_move_extracted_count
        print(f"\nCHESS_MOVE_PARSING BATCH SUMMARY:")
        print(f"   Format valid (<T></T> format): {format_valid_count}/{total_count} = {(format_valid_count/total_count):.3f}")
        print(f"   Format penalties (no </T> tag): {format_penalty_count}/{total_count} = {(format_penalty_count/total_count):.3f}")
        print(f"   Moves extracted: {move_extracted_count}/{total_count} = {(move_extracted_count/total_count):.3f}")
        print(f"   Overall accuracy: {correct_count}/{total_count} = {accuracy:.3f}")
        print()
    
    return results