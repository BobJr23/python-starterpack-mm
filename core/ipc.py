import ctypes
import mmap
import asyncio
from typing import Callable, List, Union, assert_never
from pathlib import Path
from core.state import Score, GameState, PlayerAction, Vec2
from core.conf import GameConfig, NUM_PLAYERS


HANDSHAKE_BOT = ctypes.c_uint64(0xabe119c019aaffcc)

class ProtocolId:
    HandshakeMsg = 0
    HandshakeResponse = 1
    ResetMsg = 2
    ResetResponse = 3
    TickMsg = 4
    TickResponse = 5


class HandshakeMsg(ctypes.Structure):
    _fields_ = [
        ("team", ctypes.c_uint8),
        ("config", GameConfig)
    ]
HandshakeResponse = ctypes.c_uint64

ResetMsg = Score
ResetResponse = Vec2 * NUM_PLAYERS

TickMsg = GameState
TickResponse = PlayerAction * NUM_PLAYERS

class ProtocolUnion(ctypes.Union):
    _fields_ = [
        ("handshake_msg", HandshakeMsg),
        ("handshake_response", HandshakeResponse),
        ("reset_msg", ResetMsg),
        ("reset_response", ResetResponse),
        ("tick_msg", TickMsg),
        ("tick_response", TickResponse),
    ]

class Protocol(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint8),
        ("data", ProtocolUnion),
    ]

class Shm(ctypes.Structure):
    _fields_ = [
        ("sync", ctypes.c_uint8),
        ("protocol", Protocol),
    ]


class Strategy:
    def __init__(
            self,
            on_reset: Callable[[Score], List[Vec2]],
            on_tick: Callable[[GameState], List[PlayerAction]]
    ):
        self.on_reset = on_reset
        self.on_tick = on_tick


async def poll(shm_view: Shm, expected_status: int):
    i = 0
    while True:
        if shm_view.sync == expected_status:
            return

        if i < 100:
            pass
        elif i < 1000:
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(i / 10000.0)
        i += 1


class EngineStatus:
    Ready = 0
    Busy = 1
    Finished = 2

config: None | GameConfig
team: None | int

def get_config() -> GameConfig:
    global config
    assert config is not None
    return config

def get_real_team() -> int:
    global team
    assert team is not None
    return team


class EngineChannel:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.file = open(self.path, "r+b")
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_WRITE)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "EngineChannel":
        return cls(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file'):
            self.file.close()

    async def handle_handshake(self):
        shm = Shm.from_buffer(self.mmap, 0)
        await poll(shm, EngineStatus.Ready)

        assert shm.protocol.type == ProtocolId.HandshakeMsg

        global config, team
        team_value = shm.protocol.data.handshake_msg.team
        config = GameConfig()
        ctypes.pointer(config)[0] = shm.protocol.data.handshake_msg.config

        # Store the team value globally
        team = team_value

        shm.protocol.data.handshake_response = HANDSHAKE_BOT
        shm.protocol.type = ProtocolId.HandshakeResponse
        shm.sync = EngineStatus.Busy

        return team_value


    async def handle_msg(self, strategy: Strategy):
        shm = Shm.from_buffer(self.mmap, 0)

        await poll(shm, EngineStatus.Ready)

        match shm.protocol.type:
            case ProtocolId.ResetMsg:
                score = shm.protocol.data.reset_msg
                response = strategy.on_reset(score)
                shm.protocol.data.reset_response = (Vec2 * NUM_PLAYERS)()
                for i in range(min(len(response), NUM_PLAYERS)):
                    shm.protocol.data.reset_response[i] = response[i]
            case ProtocolId.TickMsg:
                msg = shm.protocol.data.tick_msg
                response = strategy.on_tick(msg)
                shm.protocol.data.tick_response = (PlayerAction * NUM_PLAYERS)()
                for i in range(min(len(response), NUM_PLAYERS)):
                    shm.protocol.data.tick_response[i] = response[i]
            case _ as unreachable:
                assert_never(unreachable)

        shm.protocol.type += 1
        shm.sync = EngineStatus.Busy
