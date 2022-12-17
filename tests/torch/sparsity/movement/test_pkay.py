from enum import Enum


class BuildingBlockType(Enum):
    """
    Describes type of building block for transformers-based network.
    `MHSA` type is characterized by the presence 4 FC and 2 MatMul layers.
    `FF` type is characterized by the presence 2 FC layers.
    """
    MHSA = 'MHSA'
    FF = 'FF'
    Unknown = 'unknown'

    def __eq__(self, other: 'BuildingBlockType'):
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash(self.name)


print(hash(BuildingBlockType.Unknown) == hash('unknown'))
print(hash(BuildingBlockType.MHSA) == hash('MHSA'))
