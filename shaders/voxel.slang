#ifndef VOXEL
#define VOXEL

enum Type: uint8_t {
    Grass,
    Rock,
}

struct Voxel {
    bool active;
    bool reflective;
    bool refractive;
    bool placed;
    Type type;

    uint8_t into_raw() {
        uint8_t raw = 0;
        raw |= active ? 1 : 0;
        raw |= reflective ? 2 : 0;
        raw |= refractive ? 4 : 0;
        raw |= placed ? 8 : 0;
        //raw |= (uint8_t)type << 4;
        return raw;
    }

    static Voxel from_raw(uint8_t raw) {
        Voxel voxel;
        voxel.active = (raw & 1) == 1;
        voxel.reflective = ((raw >> 1) & 1) == 1;
        voxel.refractive = ((raw >> 2) & 1) == 1;
        voxel.placed = ((raw >> 3) & 1) == 1;
        //voxel.type = (Type)((raw >> 8) & 0xF);
        return voxel;
    }
}

#endif