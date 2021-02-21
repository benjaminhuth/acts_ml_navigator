#include <QApplication>

#include <spdlog/spdlog.h>

#include "InteractiveTrackViewer.hpp"

int main(int argc, char *argv[])
{
    spdlog::set_level(spdlog::level::debug);
    
    QApplication app(argc, argv);
    InteractiveTrackViewer track_viewer;
    track_viewer.show();

    return app.exec();
}
