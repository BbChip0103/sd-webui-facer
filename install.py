import launch

needs_install = False

try:
    import facer
    if facer.torch.__version_:
        needs_install = True
except ImportError:
    needs_install = True


# if needs_install:
#     launch.run_pip(f"install clip-interrogator=={CI_VERSION}", "requirements for CLIP Interrogator")


if needs_install:
    import launch
    import os
    import pkg_resources

    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

    with open(req_file) as file:
        for package in file:
            try:
                package = package.strip()
                if '==' in package:
                    package_name, package_version = package.split('==')
                    installed_version = pkg_resources.get_distribution(package_name).version
                    if installed_version != package_version:
                        launch.run_pip(f"install {package}", f"sd-webui-facer requirement: changing {package_name} version from {installed_version} to {package_version}")
                elif not launch.is_installed(package):
                    launch.run_pip(f"install {package}", f"sd-webui-facer requirement: {package}")
            except Exception as e:
                print(e)
                print(f'Warning: Failed to install {package}, some preprocessors may not work.')