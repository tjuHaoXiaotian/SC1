import argparse
import torchcraft as tc
import torchcraft.Constants as tcc
print(tcc)

parser = argparse.ArgumentParser(
    description='Plays simple micro battles with an attack closest heuristic')
parser.add_argument('-t', '--hostname', type=str,
                    help='Hostname where SC is running', default='127.0.0.1')
                    # help='Hostname where SC is running', default='172.23.22.16')
parser.add_argument('-p', '--port', default=11111,
                    help="Port to use")
parser.add_argument('-d', '--debug', default=1, type=int, help="Debug level")

args = parser.parse_args()

DEBUG = args.debug


def dprint(msg, level):
    if DEBUG > level:
        print(msg)


def get_closest(x, y, units):
    dist = float('inf')
    u = None
    for unit in units:
        d = (unit.x - x) ** 2 + (unit.y - y) ** 2
        if d < dist:
            dist = d
            u = unit
    return u


maps = [
    'Maps/BroodWar/micro/dragoons_zealots.scm',
    # 'Maps/BroodWar/micro/m5v5_c_far.scm'
]

skip_frames = 7
nrestarts = 0
total_battles = 0

while total_battles < 40:
    print("")
    print("CTRL-C to stop")
    print("")

    battles_won = 0
    battles_game = 0
    frames_in_battle = 1
    nloop = 1  # 每局的步数
    nrestarts += 1 #

    cl = tc.Client()
    cl.connect(args.hostname, args.port)
    state = cl.init(micro_battles=True)
    for pid, player in state.player_info.items():
        print("player {} named {} is {}".format(player.id, player.name,
                                                tc.Constants.races._dict[player.race]))



    # Initial setup the game
    cl.send([
        [tcc.set_map, maps[nrestarts % len(maps)], 0],
        [tcc.set_combine_frames, skip_frames],
        # [tcc.set_speed, 30],
        [tcc.set_speed, 15],
        [tcc.set_gui, 1],
        [tcc.set_cmd_optim, 1],
    ])
    while True:
        nloop += 1
        state = cl.recv()
        dprint("Received state: " + str(state), 1)

        actions = []
        print(state.deaths)
        if state.game_ended:
            dprint("GAME ENDED", 0)
            break
        elif state.battle_just_ended:
            dprint("BATTLE ENDED", 1)
            if state.battle_won:
                dprint("BATTLE WIN", 0)
                battles_won += 1
            else:
                dprint("BATTLE LOSS", 0)
            battles_game += 1
            total_battles += 1
            frames_in_battle = 0
            if battles_game >= 10:
                print("reload: ==================================================", battles_game)
                actions = [
                    [tcc.set_map, maps[nrestarts % len(maps)], 0],
                    # [tcc.quit],
                    [tcc.restart],
                ]
                print(maps[nrestarts % len(maps)])
        elif state.waiting_for_restart:
            dprint("WAITING FOR RESTART", 0)
        else:
            myunits = state.units[0]
            enemyunits = state.units[1]
            print('me:', len(myunits))
            print('enemy:', len(enemyunits))
            # if state.battle_frame_count % skip_frames == 0:
            for unit in myunits:
                target = get_closest(unit.x, unit.y, enemyunits)
                if target is not None:
                    actions.append([
                        tcc.command_unit_protected,
                        unit.id,
                        tcc.unitcommandtypes.Attack_Unit,
                        target.id,
                    ])
        if frames_in_battle > 2 * 60 * 24:
        # if frames_in_battle > 2:
            actions = [[tcc.quit]]
            nrestarts += 1
        dprint("Sending actions: " + str(actions), 1)
        cl.send(actions)
    cl.close()
