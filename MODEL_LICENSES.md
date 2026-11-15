# Model licenses

## Real-time detection checkpoint

The user-supplied checkpoint is distributed as `assets/realtime_weapons.pt`.
Its embedded metadata identifies an Ultralytics YOLO26s detection model and
declares the model license as **AGPL-3.0**. Its seven class labels are
`Blunt_Weapon`, `Explosive`, `Fire_Smoke`, `Firearm`, `Melee_Weapon`,
`Person`, and `Tool`.

The checkpoint SHA-256 is
`c6d92f38686400b90c441f35d89fb855ad46296dd51a891ba0ec1cb384796576`.
The app verifies this hash before loading bundled or downloaded weights.
This notice records the metadata found in the supplied checkpoint; it is not
legal advice and does not replace the license terms that apply to the model
or the application.
