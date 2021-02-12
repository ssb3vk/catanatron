import re
import traceback
import time
from collections import defaultdict

import click
import termplotlib as tpl
import numpy as np
import pandas as pd
import tensorflow as tf

from catanatron.models.enums import BuildingType, Resource
from catanatron_server.database import save_game_state
from catanatron.game import Game
from catanatron.models.player import RandomPlayer, Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from experimental.machine_learning.players.reinforcement import (
    QRLPlayer,
    TensorRLPlayer,
    VRLPlayer,
    PRLPlayer,
    hot_one_encode_action,
)
from experimental.machine_learning.players.mcts import MCTSPlayer
from experimental.machine_learning.features import (
    create_sample,
    get_feature_ordering,
)
from experimental.machine_learning.board_tensor_features import (
    CHANNELS,
    HEIGHT,
    WIDTH,
    create_board_tensor,
)
from experimental.machine_learning.utils import (
    get_discounted_return,
    get_tournament_return,
    get_victory_points_return,
    populate_matrices,
    DISCOUNT_FACTOR,
)


PLAYER_REGEX = re.compile("(R|W|M[0-9]+|V.*|P.*|Q.*|T.*)")


@click.command()
@click.option("-n", "--num", default=5, help="Number of games to play.")
@click.option(
    "--players",
    default="R,R,R,R",
    help="""
        Comma-separated 4 players to use. R=Random, W=WeightedRandom, VX=VRLPlayer(Version X),
            PX=PRLPlayer(Version X), QX=QRLPlayer(Version X). X >= 1.
    """,
)
@click.option(
    "-o",
    "--outpath",
    default=None,
    help="Path where to save ML csvs.",
)
def simulate(num, players, outpath):
    """Simple program simulates NUM Catan games."""
    player_keys = players.split(",")
    assert len(player_keys) == 4, "Must specify 4 players."
    assert all([PLAYER_REGEX.match(x) for x in player_keys]), "Invalid --players"

    initialized_players = []
    colors = [c for c in Color]
    pseudonyms = ["Foo", "Bar", "Baz", "Qux"]
    for i, key in enumerate(player_keys):
        player_type = key[0]
        param = key[1:]
        if player_type == "R":
            initialized_players.append(RandomPlayer(colors[i], pseudonyms[i]))
        elif player_type == "W":
            initialized_players.append(WeightedRandomPlayer(colors[i], pseudonyms[i]))
        elif player_type == "V":
            initialized_players.append(VRLPlayer(colors[i], pseudonyms[i], param))
        elif player_type == "Q":
            initialized_players.append(QRLPlayer(colors[i], pseudonyms[i], param))
        elif player_type == "P":
            initialized_players.append(PRLPlayer(colors[i], pseudonyms[i], param))
        elif player_type == "T":
            initialized_players.append(TensorRLPlayer(colors[i], pseudonyms[i], param))
        elif player_type == "M":
            initialized_players.append(MCTSPlayer(colors[i], pseudonyms[i], int(param)))
        else:
            raise ValueError("Invalid player key")

    play_batch(num, initialized_players, outpath)


def play_batch(num_games, players, games_directory):
    """Plays num_games, saves final game in database, and populates data/ matrices"""
    wins = defaultdict(int)
    turns = []
    durations = []
    games = []
    branching_factors = []
    for i in range(num_games):
        for player in players:
            player.restart_state()

        print(f"Playing game {i} / {num_games}:", players)
        if games_directory:
            data = defaultdict(
                lambda: {
                    "samples": [],
                    "actions": [],
                    "board_tensors": [],
                    # These are for practicing ML with simpler problems
                    "OWS_ONLY_LABEL": [],
                    "OWS_LABEL": [],
                    "settlements": [],
                    "cities": [],
                    "prod_vps": [],
                }
            )
            action_callback = build_action_callback(data)
            game, duration = play_and_time(players, action_callback)
            if game.winning_player() is not None:
                flush_to_matrices(game, data, games_directory)
        else:
            game, duration = play_and_time(players, None)
        branching_factors.extend(game.branching_factors)
        print("Took", duration, "seconds")
        print({str(p): p.actual_victory_points for p in players})
        save_game_state(game)
        print("Saved in db. See result at http://localhost:3000/games/" + game.id)
        print("")

        winner = game.winning_player()
        wins[str(winner)] += 1
        turns.append(game.num_turns)
        durations.append(duration)
        games.append(game)

    print("Branching Factor Stats:")
    print(pd.Series(branching_factors).describe())
    print("AVG Turns:", sum(turns) / len(turns))
    print("AVG Duration:", sum(durations) / len(durations))
    # Print Winners graph in command line:
    fig = tpl.figure()
    fig.barh([wins[str(p)] for p in players], players, force_ascii=False)
    fig.show()

    return games


def play_and_time(players, action_callback):
    game = Game(players)
    start = time.time()
    try:
        game.play(action_callback)
    except Exception as e:
        traceback.print_exc()
    finally:
        duration = time.time() - start
        return game, duration


def build_action_callback(data):
    def action_callback(game: Game):
        if len(game.actions) == 0:
            return

        action = game.actions[-1]
        player = game.players_by_color[action.color]
        data[player.color]["samples"].append(create_sample(game, player))
        data[player.color]["actions"].append(hot_one_encode_action(action))

        flattened_tensor = tf.reshape(
            create_board_tensor(game, player), (WIDTH * HEIGHT * CHANNELS,)
        ).numpy()
        data[player.color]["board_tensors"].append(flattened_tensor)

        player_tiles = set()
        for node_id in (
            player.buildings[BuildingType.SETTLEMENT]
            + player.buildings[BuildingType.CITY]
        ):
            for tile in game.board.get_adjacent_tiles(node_id):
                player_tiles.add(tile.resource)
        data[player.color]["OWS_ONLY_LABEL"].append(
            player_tiles == set([Resource.ORE, Resource.WHEAT, Resource.SHEEP])
        )
        data[player.color]["OWS_LABEL"].append(
            Resource.ORE in player_tiles
            and Resource.WHEAT in player_tiles
            and Resource.SHEEP in player_tiles
        )
        data[player.color]["settlements"].append(
            len(player.buildings[BuildingType.SETTLEMENT])
        )
        data[player.color]["cities"].append(len(player.buildings[BuildingType.CITY]))
        data[player.color]["prod_vps"].append(
            len(player.buildings[BuildingType.SETTLEMENT])
            + len(player.buildings[BuildingType.CITY])
        )

    return action_callback


def flush_to_matrices(game, data, games_directory):
    print("Flushing to matrices...")
    t1 = time.time()
    samples = []
    actions = []
    board_tensors = []
    labels = []
    for player in game.players:
        player_data = data[player.color]
        samples.extend(player_data["samples"])
        actions.extend(player_data["actions"])
        board_tensors.extend(player_data["board_tensors"])

        # Make matrix of (RETURN, DISCOUNTED_RETURN, TOURNAMENT_RETURN, DISCOUNTED_TOURNAMENT_RETURN)
        episode_return = get_discounted_return(game, player, 1)
        discounted_return = get_discounted_return(game, player, DISCOUNT_FACTOR)
        tournament_return = get_tournament_return(game, player, 1)
        vp_return = get_victory_points_return(game, player)
        discounted_tournament_return = get_tournament_return(
            game, player, DISCOUNT_FACTOR
        )
        return_matrix = np.tile(
            [
                [
                    episode_return,
                    discounted_return,
                    tournament_return,
                    discounted_tournament_return,
                    vp_return,
                ]
            ],
            (len(player_data["samples"]), 1),
        )
        return_matrix = np.concatenate(
            (return_matrix, np.transpose([player_data["OWS_ONLY_LABEL"]])), axis=1
        )
        return_matrix = np.concatenate(
            (return_matrix, np.transpose([player_data["OWS_LABEL"]])), axis=1
        )
        return_matrix = np.concatenate(
            (return_matrix, np.transpose([player_data["settlements"]])), axis=1
        )
        return_matrix = np.concatenate(
            (return_matrix, np.transpose([player_data["cities"]])), axis=1
        )
        return_matrix = np.concatenate(
            (return_matrix, np.transpose([player_data["prod_vps"]])), axis=1
        )
        labels.extend(return_matrix)

    # Build Q-learning Design Matrix
    samples_df = pd.DataFrame.from_records(
        samples, columns=get_feature_ordering()  # this must be in sync with features.
    ).astype("float64")
    board_tensors_df = pd.DataFrame(board_tensors).astype("float64")
    actions_df = pd.DataFrame(actions).astype("float64").add_prefix("ACTION_")
    rewards_df = pd.DataFrame(
        labels,
        columns=[
            "RETURN",
            "DISCOUNTED_RETURN",
            "TOURNAMENT_RETURN",
            "DISCOUNTED_TOURNAMENT_RETURN",
            "VICTORY_POINTS_RETURN",
            "OWS_ONLY_LABEL",
            "OWS_LABEL",
            "settlements",
            "cities",
            "prod_vps",
        ],
    ).astype("float64")
    print(rewards_df.describe())

    print(
        "Collected DataFrames. Data size:",
        "Samples:",
        samples_df.shape,
        "Board Tensors:",
        board_tensors_df.shape,
        "Actions:",
        actions_df.shape,
        "Rewards:",
        rewards_df.shape,
    )
    populate_matrices(
        samples_df, board_tensors_df, actions_df, rewards_df, games_directory
    )
    print("Saved to matrices at:", games_directory, ". Took", time.time() - t1)
    return samples_df, board_tensors_df, actions_df, rewards_df


if __name__ == "__main__":
    simulate()