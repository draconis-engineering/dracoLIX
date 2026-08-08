
module linalg
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    implicit none (type, external)
    private
    public :: matmatmul_c

contains

    ! ------------- CORE FUNCTIONS ------------- !

    function matmatmul(a, b) result(res)
        real(c_double), intent(in) :: a(:, :), b(:, :)
        real(c_double) :: res(size(a, 1), size(b, 2))
        integer(c_int) :: i, j, k

        do i = 1, size(a, 1)
            do j = 1, size(b, 2)
                res(i, j) = 0.0
                do k = 1, size(a, 2)
                    res(i, j) = res(i, j) + a(i, k) * b(k, j)
                end do
            end do
        end do
    end function matmatmul

    function vecmatmul(a, b) result(res)
        real, intent(in) :: a(:), b(:, :)
        real :: res(size(b, 1))
        integer :: i, j

        do i = 1, size(b, 1)
            res(i) = 0.0
            do j = 1, size(a)
                res(i) = res(i) + a(j) * b(i, j)
            end do
        end do
    end function vecmatmul

    function matvecmul(a, b) result(res)
        real, intent(in) :: a(:, :), b(:)
        real :: res(size(a, 1))
        integer :: i, j

        do i = 1, size(a, 1)
            res(i) = 0.0
            do j = 1, size(a, 2)
                res(i) = res(i) + a(i, j) * b(j)
            end do
        end do
    end function matvecmul

    function dot(a, b) result(res)
        real, intent(in) :: a(:), b(:)
        real :: res
        integer :: i

        res = 0.0
        do i = 1, size(a)
            res = res + a(i) * b(i)
        end do
    end function dot

    function outer(a, b) result(res)
        real, intent(in) :: a(:), b(:)
        real :: res(size(a), size(b))
        integer :: i, j

        do i = 1, size(a)
            do j = 1, size(b)
                res(i, j) = a(i) * b(j)
            end do
        end do
    end function outer

    ! ------------- SUBROUTINE WRAPPERS ------------- !

    subroutine matmatmul_c(a, b, res, n, m, p) bind(C, name="matmatmul_c")
        use, intrinsic :: iso_c_binding, only: c_double, c_int
        implicit none (type, external)

        integer(c_int), intent(in), value :: n, m, p
        real(c_double), intent(in) :: a(n, m), b(m, p)
        real(c_double), intent(out) :: res(n, p)

        res = matmatmul(a, b)

    end subroutine matmatmul_c

    subroutine vecmatmul_c(a, b, res, n) bind(C, name="vecmatmul_c")
        use, intrinsic :: iso_c_binding, only: c_double, c_int
        implicit none (type, external)

        integer(c_int), intent(in), value :: n ! Dimension of the vectors
        real(c_double), intent(in) :: a(n), b(n)
        real(c_double), intent(out) :: res(n)

        res = vecmatmul(a, b)

    end subroutine vecmatmul_c

    subroutine matvecmul_c(a, b, res, n, m) bind(C, name="matvecmul_c")
        use, intrinsic :: iso_c_binding, only: c_double, c_int
        implicit none (type, external)

        integer(c_int), intent(in), value :: n, m
        real(c_double), intent(in) :: a(n, m), b(n)
        real(c_double), intent(out) :: res(m)

        res = matvecmul(a, b)

    end subroutine matvecmul_c

    subroutine dot_c(a, b, res, n) bind(C, name="dot_c")
        use, intrinsic :: iso_c_binding, only: c_double, c_int
        implicit none (type, external)

        integer(c_int), intent(in), value :: n
        real(c_double), intent(in) :: a(n), b(n)
        real(c_double), intent(out) :: res

        res = dot(a, b)

    end subroutine dot_c

    subroutine outer_c(a, b, res, n) bind(C, name="outer_c")
        use, intrinsic :: iso_c_binding, only: c_double, c_int
        implicit none (type, external)

        integer(c_int), intent(in), value :: n
        real(c_double), intent(in) :: a(n), b(n)
        real(c_double), intent(out) :: res(n, n)

        res = outer(a, b)

    end subroutine outer_c
end module linalg
