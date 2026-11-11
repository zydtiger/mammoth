{
  description = "Project-independent infrastructure for AI workloads";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
  };

  outputs =
    {
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      # Dependencies come from uv.lock, so the Nix build installs exactly what
      # `uv sync` does.
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      mkPackage =
        pkgs:
        let
          pythonSet =
            (pkgs.callPackage pyproject-nix.build.packages {
              python = pkgs.python3;
            }).overrideScope
              (
                lib.composeManyExtensions [
                  pyproject-build-systems.overlays.wheel
                  (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
                ]
              );
          inherit (pkgs.callPackage pyproject-nix.build.util { }) mkApplication;
        in
        (mkApplication {
          venv = pythonSet.mkVirtualEnv "mammoth-env" { mammoth = [ "monitor" ]; };
          package = pythonSet.mammoth;
        }).overrideAttrs
          (old: {
            # Hosts install these systemd units instead of writing their own.
            passthru = (old.passthru or { }) // {
              deployUnits = ./deploy;
            };
            doInstallCheck = true;
            installCheckPhase = ''
              runHook preInstallCheck
              "$out/bin/mammoth" --version
              "$out/bin/mammoth" queue --help > /dev/null
              "$out/bin/mammoth" monitor serve --help > /dev/null
              runHook postInstallCheck
            '';

            meta = {
              description = "Project-independent infrastructure for AI workloads";
              homepage = "https://github.com/zydtiger/mammoth";
              license = lib.licenses.mit;
              platforms = lib.platforms.unix;
              mainProgram = "mammoth";
            };
          });
    in
    {
      packages = lib.genAttrs systems (
        system:
        let
          mammoth = mkPackage nixpkgs.legacyPackages.${system};
        in
        {
          inherit mammoth;
          default = mammoth;
        }
      );
    };
}
