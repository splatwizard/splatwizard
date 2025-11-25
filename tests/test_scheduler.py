from splatwizard.scheduler import Scheduler


def test_scheduler():
    s = Scheduler()

    s.register_task(range(0, 10, 1), task=lambda: None)

    s.init()


    for i in range(20):
        s.exec_task(None)
        s.step()
