# Simulation Assets

```text
assets/
├── robots/              # Shared robot models and meshes
├── common/              # Shared scene resources, such as textures
└── tasks/
    ├── water_plant/      # Task scene, object models, meshes, and textures
    ├── bimanual_hanoi/   # Scene, base, and three disks
    ├── click_mouse/      # Scene, mouse, and articulated display meshes
    ├── fold_glasses/     # Scene, articulated glasses, and case
    ├── pinch_tongs/      # Scene and articulated tongs meshes
    └── pick_bucket/      # Scene, articulated bucket, and boxed food
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

Keep the DexJoCo license only in the repository root `LICENSE`; the wheel includes
that file once. Preserve independent upstream licenses alongside third-party
assets, such as the Franka and Allegro models. When moving or renaming resources,
update their XML references.
