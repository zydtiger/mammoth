# Home Manager module: installs Mammoth and runs the queue service and its web
# monitor from the systemd user units in deploy/.
{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.services.mammoth;
  units = cfg.package.deployUnits;
  # The store path makes a package update restart the services.
  path = "PATH=${cfg.package}/bin:${config.home.profileDirectory}/bin:/usr/bin:/bin";
  monitorArgs = lib.concatStringsSep " " (
    lib.optionals (cfg.monitor.host != null) [
      "--host"
      cfg.monitor.host
    ]
    ++ lib.optional cfg.monitor.allowQueueWrites "--allow-queue-writes"
  );
  linkUnit = name: lib.nameValuePair "systemd/user/${name}" { source = "${units}/${name}"; };
in
{
  options.services.mammoth = {
    enable = lib.mkEnableOption "the Mammoth GPU job queue and its web monitor";

    package = lib.mkOption {
      type = lib.types.package;
      description = "Mammoth package that provides the executable and the deploy units.";
    };

    monitor = {
      host = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "192.0.2.10";
        description = "Address the web monitor binds. Unset keeps the monitor's own loopback default.";
      };
      allowQueueWrites = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = ''
          Whether the monitor may cancel jobs and change priorities. Enable it
          only when every client that can reach the bound address is trusted.
        '';
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      (lib.hm.assertions.assertPlatform "services.mammoth" pkgs lib.platforms.linux)
    ];

    home.packages = [ cfg.package ];

    # deploy/ owns the units; this module installs them and adapts them to the host.
    xdg.configFile =
      lib.listToAttrs (
        map linkUnit [
          "mammoth.target"
          "mammoth-queue.service"
          "mammoth-monitor.service"
        ]
      )
      // {
        "systemd/user/default.target.wants/mammoth.target".source = "${units}/mammoth.target";
        "systemd/user/mammoth-queue.service.d/host.conf".text = ''
          [Service]
          Environment=${path}
        '';
        "systemd/user/mammoth-monitor.service.d/host.conf".text = ''
          [Service]
          Environment=${path}
        ''
        + lib.optionalString (monitorArgs != "") ''
          Environment="MAMMOTH_MONITOR_ARGS=${monitorArgs}"
        '';
      };
  };
}
