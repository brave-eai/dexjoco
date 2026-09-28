# Simulation Assets

```text
assets/
├── robots/              # Shared robot models and meshes
├── common/              # Shared scene resources, such as textures
└── tasks/
    └── water_plant/      # Task scene, object models, meshes, and textures
```

## Path Rules

Use relative paths only. Do not depend on the working directory or machine-specific paths.

For a scene XML in `tasks/<task>/`, set mesh and texture directories to the
`assets/` root:

```xml
<compiler angle="radian" autolimits="true"
          meshdir="../../" texturedir="../../" />
```

All mesh and texture `file` attributes in the assembled model are then relative
to `assets/`, including those in included XML files:

```xml
<mesh file="robots/franka_emika_panda/assets/link0.stl" />
<mesh file="tasks/water_plant/plant/visuals/model_0.obj" />
<texture name="floor" type="2d"
         file="common/textures/light-gray-floor-tile.png" />
```

`include` paths are separate: they are relative to the XML file containing the
include, and are not affected by `meshdir` or `texturedir`. For example, a scene
in `tasks/water_plant/` includes:

```xml
<include file="../../robots/panda_allegro.xml" />
<include file="plant.xml" />
<include file="spray.xml" />
```

Keep compiler settings in the scene entry XML. Included robot and object XMLs
are fragments of that scene, not independently configured models.

Preserve upstream license files alongside reused assets. When moving or
renaming resources, update their XML references.
