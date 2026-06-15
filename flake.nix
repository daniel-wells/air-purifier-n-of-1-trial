{
  description = "Nix development shell for the SCED analysis project";

  inputs = {
    # Quarto on nixos-unstable currently mismatches with Pandoc on darwin.
    # nixos-25.05 provides a compatible Quarto/Pandoc pair.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
        r = pkgs.rWrapper.override {
    packages = with pkgs.rPackages; [
    pwr
    reshape2
    ggplot2
    dplyr
    tidyr
    ];
  };
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.quarto
            pkgs.gcc
            pkgs.pkg-config
            pkgs.graphviz
            r
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
          ];

          env = {
            UV_PYTHON = "${python}/bin/python";
          };

          shellHook = ''
            # Keep uv virtualenvs outside Dropbox-backed repo directories.
            export UV_PROJECT_ENVIRONMENT="$HOME/.cache/uv/venvs/$(basename "$PWD")"
            # Force Quarto/Jupyter execution to use the uv-managed interpreter.
            export QUARTO_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"

            echo "SCED Nix dev shell ready."
            echo "uv environment: $UV_PROJECT_ENVIRONMENT"
            echo "quarto python: $QUARTO_PYTHON"
            echo "1) uv sync"
            echo "2) uv run python air_purification.py"
            echo "3) uv run quarto render index.qmd"
          '';
        };
      });
}
