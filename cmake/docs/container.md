# Getting Started: Running the LiteRT-LM Container
To get the environment up and running, follow these steps from the root
directory of the project. The process is divided into build, create, and
attach phases to ensure container persistence is handled correctly.

## 1. Build the Image
First, we'll build the image using the configuration in the cmake/ directory.
This might take a moment if it's your first time, as it pulls in our build
dependencies.

```bash
podman build -f /path/to/repo/cmake/Containerfile -t litert_lm /path/to/repo
```
## 2. Create the Persistent Container
Instead of executing a one-off run, create a named container to preserve the
workspace state for future sessions. Using interactive mode ensures the
container is prepared for a functional terminal.

```bash
podman container create --interactive --tty --name litert_lm litert_lm:latest
```
## 3. Start and Join the Session
Finally, start the container and attach your shell to it.

```bash
podman start --attach litert_lm
```
**Quick Note:** If you exit the container and want to get back in later,
you don't need to rebuild or recreate it. Just run the podman start --attach
litert_lm command again and you'll be right back where you left off.

<br>

---
This project is licensed under the
[Apache 2.0 License.](https://github.com/google-ai-edge/LiteRT-LM/blob/main/LICENSE)