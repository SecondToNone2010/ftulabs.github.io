import json
import os
import urllib.request
import re

# Library sources and target paths
LIBS = {
    "hljs": {
        "npm_name": "highlight.js",
        "cdnjs_name": "highlight.js",
        "target_dir": "vendor/hljs"
    }
}

VERSIONS_FILE = "vendor/versions.json"

def get_latest_version(npm_name):
    url = f"https://registry.npmjs.org/{npm_name}/latest"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
        return data["version"]

def get_cdnjs_files(cdnjs_name, version):
    # Fetch file list from cdnjs for a specific version
    url = f"https://api.cdnjs.com/libraries/{cdnjs_name}/{version}"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            return data.get("files", [])
    except Exception as e:
        print(f"Error fetching cdnjs file list: {e}")
        return []

def update_lib(name, config, latest_version):
    print(f"Updating {name} to {latest_version}...")
    target_dir = config["target_dir"]
    
    # Get all available files for this version on cdnjs
    remote_files = get_cdnjs_files(config["cdnjs_name"], latest_version)
    if not remote_files:
        print(f"No files found for {name} on cdnjs.")
        return

    # Walk the local target_dir and find all files
    for root, _, filenames in os.walk(target_dir):
        for filename in filenames:
            # Skip versions.json if it is in the target dir (it's not but just in case)
            if filename == "versions.json":
                continue
                
            local_path = os.path.join(root, filename)
            rel_local_path = os.path.relpath(local_path, target_dir)
            
            # Find matching remote file
            # Priority 1: Match by relative path
            # Priority 2: Match by filename anywhere in the package
            remote_path = None
            if rel_local_path in remote_files:
                remote_path = rel_local_path
            else:
                # Search for filename match
                for rf in remote_files:
                    if os.path.basename(rf) == filename:
                        # Exclude 'es/' versions if there are others
                        if rf.startswith("es/"):
                            continue
                        remote_path = rf
                        break

            if not remote_path:
                print(f"  Warning: No remote match found for {rel_local_path}")
                continue

            url = f"https://cdnjs.cloudflare.com/ajax/libs/{config['cdnjs_name']}/{latest_version}/{remote_path}"
            print(f"  Downloading {url} to {local_path}...")
            try:
                with urllib.request.urlopen(url) as response:
                    content = response.read()
                    with open(local_path, "wb") as f:
                        f.write(content)
            except Exception as e:
                print(f"  Failed to download {url}: {e}")

def main():
    if not os.path.exists(VERSIONS_FILE):
        # Create versions file if it doesn't exist
        with open(VERSIONS_FILE, "w") as f:
            json.dump({}, f)

    with open(VERSIONS_FILE, "r") as f:
        versions = json.load(f)

    updated = False
    for lib_name, config in LIBS.items():
        current_version = versions.get(lib_name)
        latest_version = get_latest_version(config["npm_name"])
        
        # We always run the update if the user wants us to identify files, 
        # or if there is a version mismatch.
        # But we'll skip if the version already matches to be efficient.
        if current_version != latest_version:
            update_lib(lib_name, config, latest_version)
            versions[lib_name] = latest_version
            updated = True
        else:
            print(f"{lib_name} is already at the latest version ({latest_version}).")

    if updated:
        with open(VERSIONS_FILE, "w") as f:
            json.dump(versions, f, indent=2)
        print("Vendor versions updated.")
    else:
        print("No updates found.")

if __name__ == "__main__":
    main()
